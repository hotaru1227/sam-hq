import cv2
import os
import numpy as np
from stats_utils import get_dice_1,get_fast_aji_plus,get_fast_pq,get_fast_aji
from utils.postproc_other import process
from scipy.io import loadmat

# 文件夹路径
folder1_path = "/data/hotaru/projects/sam-hq/train/save_output/mask"
folder2_path = "/data/hotaru/projects/sam-hq/train/save_output/contour"
save_path = "/data/hotaru/projects/sam-hq/train/save_output/mask-contour"
gt_path = "/data/hotaru/projects/sam-hq/data/cpm17/test/Labels/"

# 加载图像并处理
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
    return images

# 从两个文件夹加载图像
images_folder1 = load_images(folder1_path)
images_folder2 = load_images(folder2_path)

# 创建新的图像
for idx, (image1, image2) in enumerate(zip(images_folder1, images_folder2)):
    # 确保图像尺寸相同
    assert image1.shape == image2.shape, f"Images at index {idx} have different dimensions."
    print(idx,"*"*20)
    # 创建新图像，初始化为图像1
    new_image = image1-image2
    new_image[new_image < 0] = 0
    
    # 保存新图像
    save_filename = f"image_{idx:02d}.png"
    cv2.imwrite(os.path.join(save_path, save_filename), new_image)
    
# 处理完毕
