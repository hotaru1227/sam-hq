# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from scipy.optimize import linear_sum_assignment

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        
        image_embeddings, interm_embeddings = self.image_encoder(input_images)
        
        pil_image = transforms.ToPILImage()(batched_input[0]['image'])  # 原始图像去除批次维度并转换为 PIL 图像
        tmp_save_path = "/data/hotaru/projects/sam-hq/train/segment_anything_training/tmp_output/"


        reference_img = Image.open("/data/hotaru/projects/sam-hq/data/cpm17/reference_08/cropped_image.jpg") 
        reference_mask = Image.open("/data/hotaru/projects/sam-hq/data/cpm17/reference_08/cropped_binary.png") 
        
        
        np_mask = np.array(reference_mask) 
        tensor_mask = torch.from_numpy(np_mask).unsqueeze(0).unsqueeze(0) # (1, 1, 1024, 1024)

        # 使用 torchvision.transforms 进行 resize
        resize_transform = transforms.Resize((64, 64))
        resized_tensor_mask = resize_transform(tensor_mask).squeeze(0).squeeze(0)  # 变为 (64, 64)
        flattened_tensor_mask = resized_tensor_mask.view(-1) #4096
        

        np_reference_img = np.array(reference_img)   # 将PIL Image对象转换为numpy数组，并指定数据类型为uint8（0-255）  
        tensor_reference_img= torch.from_numpy(np_reference_img.astype(np.float32) / 255.0)  # 如果需要，将numpy数组中的值缩放到0-1的范围（PyTorch通常期望输入是浮点数且范围在0-1）  
        tensor_reference_img = (tensor_reference_img * 5) - 2
        # 添加批次维度，并调整维度顺序以匹配PyTorch的NCHW（批次大小, 通道, 高度, 宽度）格式  
        # 注意：PyTorch期望通道数在第一个维度之后，但在numpy数组中是最后一个维度  
        tensor_reference_img = tensor_reference_img.permute(2, 0, 1).unsqueeze(0).to(torch.float32)    # NCHW  
        reference_embedding,_ = self.image_encoder(tensor_reference_img.cuda())

        image_embeddings_flat = image_embeddings.view(1,256, -1)[0]
        reference_embedding_flat = reference_embedding.view(1,256, -1)[0]
        
        S = reference_embedding_flat.t() @ image_embeddings_flat # ns*N, N
        C = (1 - S) / 2  # distance

        S_forward = S[flattened_tensor_mask.flatten().bool()] # n,4096
        

        indices_forward = linear_sum_assignment(S_forward.cpu().numpy(), maximize=True)
        indices_forward = [torch.as_tensor(index, dtype=torch.int64, device=self.device) for index in indices_forward]
        sim_scores_f = S_forward[indices_forward[0], indices_forward[1]]
        indices_mask = flattened_tensor_mask.flatten().nonzero()[:, 0]
        
        # S_reverse = S.t()[indices_forward[1]]
        # indices_reverse = linear_sum_assignment(S_reverse.cpu(), maximize=True)
        # indices_reverse = [torch.as_tensor(index, dtype=torch.int64, device=self.device) for index in indices_reverse]
        # retain_ind = torch.isin(indices_reverse[1].cpu(), indices_mask)
        # if not (retain_ind == False).all().item():
        #     indices_forward = [indices_forward[0][retain_ind], indices_forward[1][retain_ind]]
        #     sim_scores_f = sim_scores_f[retain_ind]
        # inds_matched, sim_matched = indices_forward, sim_scores_f

        topk_values, topk_indices = torch.topk(sim_scores_f, 256)
        selected_rows = torch.index_select(S_forward, 0, topk_indices)

        S_forward_selected_reshaped = selected_rows.view(-1, 64, 64)
        S_forward_selected_embedding = S_forward_selected_reshaped.unsqueeze(0)

        # 可视化大集合
        # pil_image.save(tmp_save_path+"input_images.jpg") #原始图像
        # mask_resized_image = Image.fromarray((resized_tensor_mask.numpy()))
        # mask_resized_image.save(tmp_save_path+'mask_resized_image.jpg')

        # S_forward_reshaped_image_0 = Image.fromarray((S_forward_reshaped[0].cpu().numpy()*255).astype(np.uint8))
        # S_forward_reshaped_image_0.save(tmp_save_path+"S_forward_reshaped_image[0].jpg")
        # S_forward_reshaped_image_1 = Image.fromarray((S_forward_reshaped[1].cpu().numpy()*255).astype(np.uint8))
        # S_forward_reshaped_image_1.save(tmp_save_path+"S_forward_reshaped_image[1].jpg")
        # S_forward_reshaped_image_2 = Image.fromarray((S_forward_reshaped[2].cpu().numpy()*255).astype(np.uint8))
        # S_forward_reshaped_image_2.save(tmp_save_path+"S_forward_reshaped_image[2].jpg")

        # def visualize_matrix(matrix, title, filename):
        #     plt.figure(figsize=(8, 6))
        #     plt.imshow(matrix.cpu(), cmap='viridis')
        #     plt.savefig(tmp_save_path+filename)
        #     plt.close()
        # visualize_matrix(image_embeddings_flat, 'Image Embeddings', 'image_embeddings.png')
        # visualize_matrix(reference_embedding_flat, 'Reference Embedding', 'reference_embedding.png')
        # visualize_matrix(S, 'Similarity Matrix S', 'similarity_matrix.png')
        # visualize_matrix(C, 'Distance Matrix C', 'distance_matrix.png')


        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold

            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "encoder_embedding": curr_embedding.unsqueeze(0),
                    "image_pe": self.prompt_encoder.get_dense_pe(),
                    "sparse_embeddings":sparse_embeddings,
                    "dense_embeddings":dense_embeddings,
                }
            )

        return outputs, interm_embeddings , S_forward_selected_embedding

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = x[:3,:,:]
        # Normalize colors
        # print("*"*12)
        # print(self.pixel_std.shape) #3,1,1
        # print(self.pixel_mean.shape)#3,1,1
        # print(x.shape)#正确应该是3,1024,1024
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
