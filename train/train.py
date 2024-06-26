# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
# torch.cuda.set_device(4)
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4"
import argparse
from skimage import measure
import numpy as np
import re
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc

import warnings
from stats_utils import get_dice_1,get_fast_aji_plus,get_fast_pq,get_fast_aji
from utils.postproc_other import process
from scipy.ndimage import label, center_of_mass
# from .utils.post_process import process
# from .utils import post_process 
# 忽略所有警告消息
warnings.simplefilter("ignore")




class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoderHQ(MaskDecoder):
    def __init__(self, model_type):
        super().__init__(transformer_dim=256,
                        transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=256,
                                mlp_dim=2048,
                                num_heads=8,
                            ),
                        num_multimask_outputs=3,
                        activation=nn.GELU,
                        iou_head_depth= 3,
                        iou_head_hidden_dim= 256,)
        assert model_type in ["vit_b","vit_l","vit_h"]
        checkpoint_dict = {"vit_b":"/data/hotaru/projects/sam-hq/pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
                           "vit_l":"/data/hotaru/projects/sam-hq/pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           'vit_h':"pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        for n,p in self.named_parameters():
            p.requires_grad = False

        transformer_dim=256
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )

        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """
        
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature = hq_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)

        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:,mask_slice]

        masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :]
        
        if hq_token_only:
            return masks_hq
        else:
            return masks_sam, masks_hq

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_ours],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred



def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)
    parser.add_argument('--gpu', default=4, type=int)
    

    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--bi_output", type=str, 
                        help="Path to the directory where binary masks will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=50, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def main(net, train_datasets, valid_datasets,test_datasets, args):

    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [
                                                                    RandomHFlip(),
                                                                    LargeScaleJitter()
                                                                    ],
                                                        batch_size = args.batch_size_train,
                                                        training = True)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    test_im_gt_list = get_im_gt_name_dict(test_datasets, flag="test")
    test_dataloaders, test_datasets = create_dataloaders(test_im_gt_list,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          training=False)
    print(len(test_dataloaders), " test dataloaders created")
    
    ### --- Step 2: DistributedDataParallel---
    if torch.cuda.is_available():
        net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    net_without_ddp = net.module

 
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        _ = sam.to(device=args.device)
        sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)

        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net_without_ddp.load_state_dict(torch.load(args.restore_model))
            else:
                net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"))
    
        evaluate(args, net, sam, test_dataloaders, args.visualize)


def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    net.train()
    _ = net.to(device=args.device)
    
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    
    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(train_dataloaders,1000):
            inputs, labels = data['image'], data['label']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            # input prompt
            input_keys = ['box','point','noise_mask']
            labels_box = misc.masks_to_boxes(labels[:,0,:,:])
            try:
                labels_points = misc.masks_sample_points(labels[:,0,:,:])
            except:
                # less than 10 points
                input_keys = ['box','noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)

            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

            masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )

            loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))
            loss = loss_mask + loss_dice
            
            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)


        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats = evaluate(args, net, sam, valid_dataloaders)
        train_stats.update(test_stats)
        
        net.train()  

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(net.module.state_dict(), args.output + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    if misc.is_main_process():
        sam_ckpt = torch.load(args.checkpoint)
        hq_decoder = torch.load(args.output + model_name)
        for key in hq_decoder.keys():
            sam_key = 'mask_decoder.'+key
            if sam_key not in sam_ckpt.keys():
                sam_ckpt[sam_key] = hq_decoder[key]
        model_name = "/sam_hq_epoch_"+str(epoch)+".pth"
        torch.save(sam_ckpt, args.output + model_name)


def origin_process(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    
    return postprocess_preds,target

def measure_process(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    post_preds = measure.label((postprocess_preds[0][0]>0).detach().cpu())
    post_preds_add_1 = torch.from_numpy(post_preds).unsqueeze(0)
    return post_preds_add_1,target

def compute_iou(preds, target):
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(preds[i],target[i])
    return iou / len(preds)


def compute_dice(preds, target ,mode = "binary"):
    if mode =="binary":
        dice = 0
        for i in range(0,len(preds)):
            dice = dice + misc.get_dice_1(target[i],preds[i])
            # print(dice)
        return dice / len(preds)
    if mode =="inst":
        dice = 0
        for i in range(0,len(preds)):
            dice = dice + misc.get_dice_2(target[i],preds[i])
            # print(dice)
        return dice / len(preds)
        

def compute_aji(preds, target,mode = "binary"):
    aji = 0
    for i in range(0,len(preds)):
        aji = aji + misc.get_fast_aji(target[i],preds[i],mode)
    return aji / len(preds)

def compute_aji_plus(preds, target,mode = "binary"):
    aji_p = 0
    for i in range(0,len(preds)):
        aji_p = aji_p + misc.get_fast_aji_plus(target[i],preds[i],mode)
    return aji_p / len(preds)

def compute_pq(preds, target,mode = "binary"):
    dq = 0
    sq = 0
    pq = 0
    for i in range(0,len(preds)):
        [dq_tmp,sq_tmp,pq_tmp],_ =  misc.get_fast_pq(target[i],preds[i],mode=mode)
        dq+=dq_tmp
        sq+=sq_tmp
        pq+=pq_tmp
    return dq / len(preds) , sq / len(preds),pq / len(preds)

def compute_boundary_iou(preds, target):
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],preds[i])
    return iou / len(preds)

def dilate_instance_map(instance_map, iterations=3):
    # 使用膨胀操作
    kernel = np.ones((3, 3), np.uint8)
    dilated_map = cv2.dilate(instance_map.astype(np.uint8), kernel, iterations=iterations)
    return dilated_map

def compare_and_color(pred_map, true_map):
    # 创建一个空白的RGB图像
    h, w = pred_map.shape
    result_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 判断像素值并赋予颜色
    for i in range(h):
        for j in range(w):
            pred_val = pred_map[i, j]
            true_val = true_map[i, j]

            if pred_val == 0 and true_val == 0:
                result_img[i, j] = [0, 0, 0]  # 黑色
            elif pred_val > 0 and true_val == 0:
                result_img[i, j] = [0, 0, 255]  # 红色
            elif pred_val == 0 and true_val > 0:
                result_img[i, j] = [255, 0, 0]  # 蓝色
            elif pred_val > 0 and true_val > 0:
                result_img[i, j] = [0, 255, 0]  # 绿色

    return result_img
def evaluate(args, net, sam, valid_dataloaders, visualize=False):
    net.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,1000):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori,inst_labels_ori = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label'],data_val['ori_inst_label'][0]

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()
                inst_labels_ori = inst_labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            # labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            # input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                # input_type = random.choice(input_keys)
                labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
                input_type = "box"
                if input_type == 'box':
                    # print("infer,box")
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    print("infer,point")
                    # point_coords = labels_points[b_i:b_i+1]
                    # dict_input['point_coords'] = point_coords
                    # dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                    point_path = "/data/hotaru/projects/sam-hq/data/cpm17/test/cpm17_test_gt"
                    pattern = r'image_(\d+)\.png'
                    # 使用正则表达式匹配字符串
                    ori_gt_path = data_val['ori_gt_path'][0]
                    match = re.search(pattern, ori_gt_path).group(1)
                    point_path = point_path+"/image_"+match+".npy"
                    point_data = np.load(point_path)
                    # print(point_data)
                    dict_input['point_coords'] = torch.unsqueeze(torch.tensor(point_data[:, :2]) , dim=0)  #n,1,2 👉 1,n,2
                    dict_input['point_labels'] = torch.ones(dict_input['point_coords'].shape[1], device=dict_input['point_coords'].device)[None,:] #n,1 👉1,n
                    # dict_input['point_labels'] = torch.unsqueeze(torch.squeeze(dict_input['point_labels']), dim=1) 

                    # print("1")
                elif input_type == 'noise_mask':
                    print("infer,noise_mask")
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings , S_forward_selected_embedding = sam(batched_input, multimask_output=False)  # batched_output
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            tmp_embedding = dense_embeddings[0]+S_forward_selected_embedding/10
            # tmp_embedding = dense_embeddings[0]
            masks_sam, masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=tmp_embedding,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings,
            )


            
            ori_gt_path = data_val['ori_gt_path'][0]
            pattern = r'/([^/]+)\.png$'
            match = re.search(pattern, ori_gt_path).group(1)
            image_name = "image_"+match+".png"
            # '''
            # 就是看看这个质量如何 要删掉,记得膨胀像素
            # '''
            # maskSUBcontour_path = "/data/hotaru/projects/sam-hq/train/save_output/540images/mask-contour/"
            # image_name = ""+match+".png"
            # masks_hq  = cv2.imread(maskSUBcontour_path+image_name, cv2.IMREAD_GRAYSCALE)
            # masks_hq[masks_hq==1] = 0  #啊啊啊太蠢了
            # masks_hq = torch.from_numpy(masks_hq).unsqueeze(0).unsqueeze(0).cuda().float()
            # '''
            # 以上
            # '''


            # 把原本的后处理封装了一下
            origin_process_pred,_ = origin_process(masks_hq,labels_ori)
            # origin_process_pred = masks_hq
            iou = compute_iou(origin_process_pred,labels_ori)  #先看这个报不报错
            boundary_iou = compute_boundary_iou(origin_process_pred,labels_ori)
            dice = compute_dice(origin_process_pred,labels_ori)
            aji = compute_aji(origin_process_pred,labels_ori)
            aji_p = compute_aji_plus(origin_process_pred,labels_ori)
            dq, sq, pq= compute_pq(origin_process_pred,labels_ori)
            
            
            # measure_process_inst = measure_process(masks_hq,labels_ori)[0] #这种尝试过了，得到了一组结果


            measure_process_inst = process(masks_hq,labels_ori,"?") #1,600,600
            # measure_process_inst = dilate_instance_map(measure_process_inst,1) #膨胀三个像素
            measure_process_inst = relabel_instances(measure_process_inst)#发现实例id不连续，大概是数组越界报错的原因，额外加处理一步
            inst_labels_ori = relabel_instances(inst_labels_ori.cpu().numpy())
            inst_labels_ori = torch.from_numpy(inst_labels_ori)

            measure_process_inst = torch.from_numpy(measure_process_inst).unsqueeze(0)#发现实例id不连续，大概是数组越界报错的原因，额外加处理一步
            

            dice_inst = compute_dice(measure_process_inst,inst_labels_ori,"inst") #1,600,600
            aji_inst = compute_aji(measure_process_inst,inst_labels_ori,"inst")
            aji_p_inst = compute_aji_plus(measure_process_inst,inst_labels_ori,"inst")
            dq_inst, sq_inst, pq_inst = compute_pq(measure_process_inst,inst_labels_ori,"inst")

            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_hq_vis = (F.interpolate(measure_process_inst.float().unsqueeze(0).detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    # print('base:', base)
                    save_base = os.path.join(args.output, image_name)
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    show_anns(masks_hq_vis[ii], None,  None,None, save_base , imgs_ii, show_iou, show_boundary_iou)
                

                #为应用后处理额外保存的二值图mask结果
                # binary_save_path = args.bi_output+'/'
                # # plt.imshow(origin_process_pred.cpu().detach().numpy()[0][0]>0)
                # origin_process_pred_cpu = origin_process_pred.cpu().detach()
                # # 2. 转换为 NumPy 数组
                # origin_process_pred_numpy = origin_process_pred_cpu.numpy()
                # from PIL import Image
                # binary_semantic_map = np.where(origin_process_pred_numpy[0][0] > 0, 1, 0)
                # binary_semantic_map_image = Image.fromarray(binary_semantic_map.astype(np.uint8) * 255)
                # save_path = binary_save_path+str(k)+'_'+ str(base)+'.png'
                # binary_semantic_map_image.save(save_path)
                
                #这里再保存实例图 糖果随机色
                measure_process_inst_np = measure_process_inst[0].cpu().numpy()
                num_instances = np.max(measure_process_inst_np) + 1
                colors = np.random.randint(0, 255, (num_instances, 3))
                inst_img = colors[measure_process_inst_np]
                plt.imsave(args.output+"/"+image_name+"_inst.png", inst_img.astype(np.uint8))
                ####

                #保存一下差异图 在这个上面点点吧
                #point:batched_input[0]['point_coords']
                pred_map = torch.squeeze(measure_process_inst).cpu().numpy()
                true_map = torch.squeeze(inst_labels_ori).cpu().numpy()
                result_img = compare_and_color(pred_map, true_map)
                if 'point_coords' in batched_input[0].keys() and batched_input[0]['point_coords'] is not None:
                    for input_data in batched_input:
                        point_coords = input_data['point_coords']
                        for point in point_coords.numpy()[0]:
                            # 将 point_coords 中的每个坐标画成半径为 5 的白色圆点
                            cv2.circle(result_img, (point[0], point[1]), radius=3, color=(255, 255, 255), thickness=-1)

                cv2.imwrite(args.output+"/"+image_name+"_differ.png", result_img)

            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou,
            "dice_"+str(k): dice,"aji_"+str(k): aji,"aji_p_"+str(k): aji_p,"dq_"+str(k): dq,"sq_"+str(k): sq,"pq_"+str(k): pq,
                            "dice_i_"+str(k): dice_inst,"aji_i_"+str(k): aji_inst,"aji_p_i_"+str(k): aji_p_inst,"dq_i_"+str(k): dq_inst,"sq_i_"+str(k): sq_inst,"pq_i_"+str(k): pq_inst}
            # loss_dict_reduced = misc.reduce_dict(loss_dict)
            print(image_name,dice_inst,aji_inst,aji_p_inst,dq_inst,sq_inst,pq_inst)
            metric_logger.update(**loss_dict)
            

        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)


    return test_stats

def relabel_instances(inst):
    unique_ids = np.unique(inst)
    new_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    new_inst = np.zeros_like(inst)
    for old_id, new_id in new_id_map.items():
        new_inst[inst == old_id] = new_id
    return new_inst

if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------



    dataset_cpm17_origin_train = {"name": "CPM-17",
                 "im_dir": "/data/hotaru/projects/sam-hq/data/cpm17/train/Images",
                 "gt_dir": "/data/hotaru/projects/sam-hq/data/cpm17/train/Labels_binary_png", 
                #  "gt_dir": "/data/hotaru/projects/sam-hq/data/cpm17/train/Labels_binary_contour_png",
                 "inst_gt_dir":"/data/hotaru/projects/sam-hq/data/cpm17/train/Labels",
                 "im_ext": ".png",
                 "gt_ext": ".png",
                 "inst_gt_ext":".mat"}
    dataset_cpm17_origin_test = {"name": "CPM-17",
                 "im_dir": "/data/hotaru/projects/sam-hq/data/cpm17/test/Images",
                 "gt_dir": "/data/hotaru/projects/sam-hq/data/cpm17/test/Labels_binary_png",
                # "gt_dir": "/data/hotaru/projects/sam-hq/data/cpm17/test/Labels_binary_contour_png",
                 "inst_gt_dir":"/data/hotaru/projects/sam-hq/data/cpm17/test/Labels",
                 "im_ext": ".png",
                 "gt_ext": ".png",
                 "inst_gt_ext":".mat"}
    
    dataset_cpm17_patch_540_train= {"name": "CPM-17",
                 "im_dir": "/data/hotaru/projects/sam-hq/data/cpm17_540/train/Image",
                 "gt_dir": "/data/hotaru/projects/sam-hq/data/cpm17_540/train/Label_binary", 
                 "inst_gt_dir":"/data/hotaru/projects/sam-hq/data/cpm17_540/train/Label_inst",
                 "im_ext": ".png",
                 "gt_ext": ".png",
                 "inst_gt_ext":".mat"}
    dataset_cpm17_patch_540_test= {"name": "CPM-17",
                 "im_dir": "/data/hotaru/projects/sam-hq/data/cpm17_540/valid/Image",
                 "gt_dir": "/data/hotaru/projects/sam-hq/data/cpm17_540/valid/Label_binary", 
                 "inst_gt_dir":"/data/hotaru/projects/sam-hq/data/cpm17_540/valid/Label_inst",
                 "im_ext": ".png",
                 "gt_ext": ".png",
                 "inst_gt_ext":".mat"}

    train_datasets = [dataset_cpm17_patch_540_train]
    valid_datasets = [dataset_cpm17_origin_train] 
    test_datasets = [dataset_cpm17_origin_test] 

    # train_datasets = [dataset_cpm17_origin_train]
    # valid_datasets = [dataset_cpm17_origin_train] 
    # test_datasets = [dataset_cpm17_origin_test] 
    import torch.distributed as dist
    # dist.init_process_group(backend='nccl', init_method='env://')
    args = get_args_parser()
    net = MaskDecoderHQ(args.model_type) 

    main(net, train_datasets, valid_datasets,test_datasets, args)
