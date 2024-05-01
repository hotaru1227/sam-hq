# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:28:51 2020

@author: he
"""


import cv2
import numpy as np
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
                                    binary_erosion,
                                    binary_dilation, 
                                    binary_fill_holes,
                                    distance_transform_cdt,
                                    distance_transform_edt)
from skimage.morphology import remove_small_objects#, watershed
from skimage.segmentation import watershed
#import matplotlib.pyplot as plt #服务器中会报错，事先注释了 hhl20200521add
import torch.nn.functional as F

def process(pred,target,model_mode, min_size = 10, ws=True):
    def gen_inst_dst_map(ann):  
        shape = ann.shape[:2] # HW
        nuc_list = list(np.unique(ann))
        nuc_list.remove(0) # 0 is background

        canvas = np.zeros(shape, dtype=np.uint8)
        for nuc_id in nuc_list:
            nuc_map   = np.copy(ann == nuc_id)    
            nuc_dst = distance_transform_edt(nuc_map)
            nuc_dst = 255 * (nuc_dst / np.amax(nuc_dst))       
            canvas += nuc_dst.astype('uint8')
        return canvas

    assert target.shape[1] == 1, 'only support one mask per image now'
    if(pred.shape[2]!=target.shape[2] or pred.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(pred, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = pred
    pred = postprocess_preds[0][0].cpu().detach().numpy()
    if model_mode != 'dcan':
        assert len(pred.shape) == 2, 'Prediction shape is not HW'
        pred[pred  > 0.5] = 1
        pred[pred <= 0.5] = 0

        # ! refactor these
        ws = False if model_mode == 'unet' or model_mode == 'micronet' else ws
        if ws:
            dist = measurements.label(pred)[0]
            dist = gen_inst_dst_map(dist)
            marker = np.copy(dist)
            marker[marker <= 125] = 0
            marker[marker  > 125] = 1
            marker = binary_fill_holes(marker) 
            marker = binary_erosion(marker, iterations=1)
            marker = measurements.label(marker)[0]

            marker = remove_small_objects(marker, min_size=min_size)
            pred = watershed(-dist, marker, mask=pred)
            pred = remove_small_objects(pred, min_size=min_size)
            #print('============================ ws = True ============================ ')
        else:
            pred = binary_fill_holes(pred) 
            pred = measurements.label(pred)[0]
            pred = remove_small_objects(pred, min_size=min_size)
            print('binary_fill_holes(pred), measurements.label(pred)[0], remove_small_objects(pred, min_size=10)')
        
        if model_mode == 'micronet':
            # * dilate with same kernel size used for erosion during training
            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], np.uint8)
    
            canvas = np.zeros([pred.shape[0], pred.shape[1]])
            for inst_id in range(1, np.max(pred)+1):
                inst_map = np.array(pred == inst_id, dtype=np.uint8)
                inst_map = cv2.dilate(inst_map, kernel, iterations=1)
                inst_map = binary_fill_holes(inst_map)
                canvas[inst_map > 0] = inst_id
            pred = canvas
    else:
        assert (pred.shape[2]) == 2, 'Prediction should have contour and blb'
        blb = pred[...,0]
        blb = np.squeeze(blb)
        cnt = pred[...,1]
        cnt = np.squeeze(cnt)

        pred = blb - cnt # NOTE
        pred[pred  > 0.3] = 1 # Kumar 0.3, UHCW 0.3
        pred[pred <= 0.3] = 0 # CPM2017 0.1
        pred = measurements.label(pred)[0]
        pred = remove_small_objects(pred, min_size=min_size) # 20
        canvas = np.zeros([pred.shape[0], pred.shape[1]])

        k_disk = np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], np.uint8)
        for inst_id in range(1, np.max(pred)+1):
            inst_map = np.array(pred == inst_id, dtype=np.uint8)
            inst_map = cv2.dilate(inst_map, k_disk, iterations=1)
            inst_map = binary_fill_holes(inst_map)
            canvas[inst_map > 0] = inst_id
        pred = canvas
        
    return pred