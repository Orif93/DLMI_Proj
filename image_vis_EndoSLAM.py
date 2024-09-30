import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


# EndoSLAM
path_hamlyn = r'/home/orifr/ori/DLMI/proj/images_for_comparision/endoslam/'
gt1_path = os.path.join(path_hamlyn, r'colon/gt/aov_image_0015.png')
rgb_1 = os.path.join(path_hamlyn, r'colon/rgb/image_0015.png')
dav1_1 = os.path.join(path_hamlyn, 'colon/dav1/image_0015_depth.png')
dav2_1 = os.path.join(path_hamlyn, 'colon/dav2/image_0015.png')

gt_1 = cv2.imread(gt1_path,cv2.IMREAD_UNCHANGED)[:, :,:3]  # EndoSLALM - unit8 (320,320,4)
gt_1 = cv2.normalize(gt_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
rgb_1 = cv2.imread(rgb_1, cv2.IMREAD_COLOR)
rgb_1 = cv2.cvtColor(rgb_1, cv2.COLOR_BGR2RGB)
dav1_1 = cv2.imread(dav1_1, cv2.IMREAD_UNCHANGED)
dav2_1 = cv2.imread(dav2_1, cv2.IMREAD_UNCHANGED)
####
gt2_path = os.path.join(path_hamlyn, r'colon/gt/aov_image_0115.png')
rgb_2 = os.path.join(path_hamlyn, r'colon/rgb/image_0115.png')
dav1_2 = os.path.join(path_hamlyn, 'colon/dav1/image_0115_depth.png')
dav2_2 = os.path.join(path_hamlyn, 'colon/dav2/image_0115.png')

gt_2 = cv2.imread(gt2_path,cv2.IMREAD_UNCHANGED)[:, :,:3]  # EndoSLALM - unit8 (320,320,4)
gt_2 = cv2.normalize(gt_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
rgb_2 = cv2.imread(rgb_2, cv2.IMREAD_COLOR)
rgb_2 = cv2.cvtColor(rgb_2, cv2.COLOR_BGR2RGB)
dav1_2 = cv2.imread(dav1_2, cv2.IMREAD_UNCHANGED)
dav2_2 = cv2.imread(dav2_2, cv2.IMREAD_UNCHANGED)
####
gt3_path = os.path.join(path_hamlyn, r'small/gt/aov_image_0060.png')
rgb_3 = os.path.join(path_hamlyn, r'small/rgb/image_0060.png')
dav1_3 = os.path.join(path_hamlyn, 'small/dav1/image_0060_depth.png')
dav2_3 = os.path.join(path_hamlyn, 'small/dav2/image_0060.png')

gt_3 = cv2.imread(gt3_path,cv2.IMREAD_UNCHANGED)[:, :,:3]  # EndoSLALM - unit8 (320,320,4)
gt_3 = cv2.normalize(gt_3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
rgb_3 = cv2.imread(rgb_3, cv2.IMREAD_COLOR)
rgb_3 = cv2.cvtColor(rgb_3, cv2.COLOR_BGR2RGB)
dav1_3 = cv2.imread(dav1_3, cv2.IMREAD_UNCHANGED)
dav2_3 = cv2.imread(dav2_3, cv2.IMREAD_UNCHANGED)
####
gt4_path = os.path.join(path_hamlyn, r'small/gt/aov_image_0365.png')
rgb_4 = os.path.join(path_hamlyn, r'small/rgb/image_0365.png')
dav1_4 = os.path.join(path_hamlyn, 'small/dav1/image_0365_depth.png')
dav2_4 = os.path.join(path_hamlyn, 'small/dav2/image_0365.png')

gt_4 = cv2.imread(gt4_path,cv2.IMREAD_UNCHANGED)[:, :,:3]  # EndoSLALM - unit8 (320,320,4)
gt_4 = cv2.normalize(gt_4, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
rgb_4 = cv2.imread(rgb_4, cv2.IMREAD_COLOR)
rgb_4 = cv2.cvtColor(rgb_4, cv2.COLOR_BGR2RGB)
dav1_4 = cv2.imread(dav1_4, cv2.IMREAD_UNCHANGED)
dav2_4 = cv2.imread(dav2_4, cv2.IMREAD_UNCHANGED)
####
gt5_path = os.path.join(path_hamlyn, r'stomach/gt/aov_image_0026.png')
rgb_5 = os.path.join(path_hamlyn, r'stomach/rgb/image_0026.png')
dav1_5 = os.path.join(path_hamlyn, 'stomach/dav1/image_0026_depth.png')
dav2_5 = os.path.join(path_hamlyn, 'stomach/dav2/image_0026.png')

gt_5 = cv2.imread(gt5_path,cv2.IMREAD_UNCHANGED)[:, :,:3]  # EndoSLALM - unit8 (320,320,4)
gt_5 = cv2.normalize(gt_5, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
rgb_5 = cv2.imread(rgb_5, cv2.IMREAD_COLOR)
rgb_5 = cv2.cvtColor(rgb_5, cv2.COLOR_BGR2RGB)
dav1_5 = cv2.imread(dav1_5, cv2.IMREAD_UNCHANGED)
dav2_5 = cv2.imread(dav2_5, cv2.IMREAD_UNCHANGED)
####
gt6_path = os.path.join(path_hamlyn, r'stomach/gt/aov_image_0169.png')
rgb_6 = os.path.join(path_hamlyn, r'stomach/rgb/image_0169.png')
dav1_6 = os.path.join(path_hamlyn, 'stomach/dav1/image_0169_depth.png')
dav2_6 = os.path.join(path_hamlyn, 'stomach/dav2/image_0169.png')

gt_6 = cv2.imread(gt6_path,cv2.IMREAD_UNCHANGED)[:, :,:3]  # EndoSLALM - unit8 (320,320,4)
gt_6 = cv2.normalize(gt_6, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
rgb_6 = cv2.imread(rgb_6, cv2.IMREAD_COLOR)
rgb_6 = cv2.cvtColor(rgb_6, cv2.COLOR_BGR2RGB)
dav1_6 = cv2.imread(dav1_6, cv2.IMREAD_UNCHANGED)
dav2_6 = cv2.imread(dav2_6, cv2.IMREAD_UNCHANGED)
# cv2.imshow('true_depth_img', dav2_1)
# cv2.waitKey(0)

gt_images = [gt_1, gt_2, gt_3, gt_4, gt_5, gt_6]
rgb_images = [rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_6]
indoor_images = [dav1_1, dav1_2, dav1_3, dav1_4, dav1_5, dav1_6]
outdoor_images = [dav2_1, dav2_2, dav2_3, dav2_4, dav2_5, dav2_6]
# Create a 6x4 subplot grid
fig, axes = plt.subplots(6, 4, figsize=(15, 15))

# Titles for columns
titles = ['Input', 'Ground\nTruth', 'DAV1', 'DAV2']

# Plot images in the grid
for row in range(6):
    # Indexes for each type of image
    rgb_img = rgb_images[row]
    gt_img = gt_images[row]
    indoor_img = indoor_images[row]
    outdoor_img = outdoor_images[row]

    # Plot RGB images
    axes[row, 0].imshow(rgb_img)
    axes[row, 0].axis('off')

    # Plot Ground Truth images as grayscale
    axes[row, 1].imshow(gt_img, cmap='gray')
    axes[row, 1].axis('off')

    # Plot Indoor images
    axes[row, 2].imshow(indoor_img)
    axes[row, 2].axis('off')

    # Plot Outdoor images
    axes[row, 3].imshow(outdoor_img)
    axes[row, 3].axis('off')

for col in range(4):
    # Calculate horizontal position for column titles
    col_center = (col + 0.5) / 4  # Adjust position for center alignment
    fig.text(col_center, 0.02, titles[col], ha='center', va='center', fontsize=30, fontweight='bold')

# Adjust layout
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust space for titles
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(r'/home/orifr/ori/DLMI/proj/images_for_comparision/endoslam/endoslam.png', bbox_inches='tight', pad_inches=0)
plt.close()