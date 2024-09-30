import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
#Hamlyn
path_hamlyn = r'/home/orifr/ori/DLMI/proj/images_for_comparision/hamlymn/'
gt_1 = os.path.join(path_hamlyn, '01/gt/0000000000.png')
rgb_1 = os.path.join(path_hamlyn, '01/rgb/0000000000.jpg')
dav_2_indoor_1 = os.path.join(path_hamlyn, '01/dav2_indoor/0000000000.png')
dav_2_outdoor_1 = os.path.join(path_hamlyn, '01/dav2_outdoor/0000000000.png')

gt_2 = os.path.join(path_hamlyn, '01/gt/0000000868.png')
rgb_2 = os.path.join(path_hamlyn, '01/rgb/0000000868.jpg')
dav_2_indoor_2 = os.path.join(path_hamlyn, '01/dav2_indoor/0000000868.png')
dav_2_outdoor_2 = os.path.join(path_hamlyn, '01/dav2_outdoor/0000000868.png')

gt_3 = os.path.join(path_hamlyn, '04/gt/0000000000.png')
rgb_3 = os.path.join(path_hamlyn, '04/rgb/0000000000.jpg')
dav_2_indoor_3 = os.path.join(path_hamlyn, '04/dav2_indoor/0000000000.png')
dav_2_outdoor_3 = os.path.join(path_hamlyn, '04/dav2_outdoor/0000000000.png')

gt_4 = os.path.join(path_hamlyn, '04/gt/0000000869.png')
rgb_4 = os.path.join(path_hamlyn, '04/rgb/0000000869.jpg')
dav_2_indoor_4 = os.path.join(path_hamlyn, '04/dav2_indoor/0000000869.png')
dav_2_outdoor_4 = os.path.join(path_hamlyn, '04/dav2_outdoor/0000000869.png')


gt_1 = cv2.imread(gt_1, cv2.IMREAD_UNCHANGED)  # Hamlyn - unit16
gt_1 = cv2.normalize(gt_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
rgb_1 = cv2.imread(rgb_1, cv2.IMREAD_COLOR)
rgb_1 = cv2.cvtColor(rgb_1, cv2.COLOR_BGR2RGB)
indoor_1 = cv2.imread(dav_2_indoor_1, cv2.IMREAD_UNCHANGED)
outdoor_1 = cv2.imread(dav_2_outdoor_1, cv2.IMREAD_UNCHANGED)

gt_2 = cv2.imread(gt_2, cv2.IMREAD_UNCHANGED)  # Hamlyn - unit16
gt_2 = cv2.normalize(gt_2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
rgb_2 = cv2.imread(rgb_2, cv2.IMREAD_COLOR)
rgb_2 = cv2.cvtColor(rgb_2, cv2.COLOR_BGR2RGB)
indoor_2 = cv2.imread(dav_2_indoor_2, cv2.IMREAD_UNCHANGED)
outdoor_2 = cv2.imread(dav_2_outdoor_2, cv2.IMREAD_UNCHANGED)

gt_3 = cv2.imread(gt_3, cv2.IMREAD_UNCHANGED)  # Hamlyn - unit16
gt_3 = cv2.normalize(gt_3, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
rgb_3 = cv2.imread(rgb_3, cv2.IMREAD_COLOR)
rgb_3 = cv2.cvtColor(rgb_3, cv2.COLOR_BGR2RGB)
indoor_3 = cv2.imread(dav_2_indoor_3, cv2.IMREAD_UNCHANGED)
outdoor_3 = cv2.imread(dav_2_outdoor_3, cv2.IMREAD_UNCHANGED)

gt_4 = cv2.imread(gt_4, cv2.IMREAD_UNCHANGED)  # Hamlyn - unit16
gt_4 = cv2.normalize(gt_4, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
rgb_4 = cv2.imread(rgb_4, cv2.IMREAD_COLOR)
rgb_4 = cv2.cvtColor(rgb_4, cv2.COLOR_BGR2RGB)
indoor_4 = cv2.imread(dav_2_indoor_4, cv2.IMREAD_UNCHANGED)
outdoor_4 = cv2.imread(dav_2_outdoor_4, cv2.IMREAD_UNCHANGED)

gt_images = [gt_1, gt_2, gt_3, gt_4]
rgb_images = [rgb_1, rgb_2, rgb_3, rgb_4]
indoor_images = [indoor_1, indoor_2, indoor_3, indoor_4]
outdoor_images = [outdoor_1, outdoor_2, outdoor_3, outdoor_4]
# Create a 4x4 subplot grid
fig, axes = plt.subplots(4, 4, figsize=(15, 15))

# Titles for columns
titles = ['Input', 'Ground\nTruth', 'DAV2\nIndoor', 'DAV2\nOutdoor']

# Plot images in the grid
for row in range(4):
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
plt.savefig(r'/home/orifr/ori/DLMI/proj/images_for_comparision/hamlymn/hamlymn.png', bbox_inches='tight', pad_inches=0)
plt.close()