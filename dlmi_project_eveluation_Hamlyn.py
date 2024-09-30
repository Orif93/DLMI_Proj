import torch
import numpy as np
import os
import cv2
from tqdm import tqdm

abs_rel_all, sq_rel_all, rmse_all, d1_all = [] , [] , [] , []
true_depth_path = r'/home/orifr/ori/datasets/Hamlyn_Dataset/rectified04/depth01'
pred_base_dir =  r'/home/orifr/ori/DLMI/Hamlyn_Dataset/rectified04/image01'
da_model = 'large'
dav1_path = os.path.join(pred_base_dir,'dav1', da_model)
dav2_path = os.path.join(pred_base_dir,'dav2', da_model)
dav2_metric_path = os.path.join(pred_base_dir,'dav2_metric', 'indoor_large')

true_depths = [img for img in os.listdir(true_depth_path) if img.endswith(".png")]
true_depths = np.sort(true_depths)

pred_depths_dav2_metric_npy =  [img[:-4] for img in os.listdir(dav2_metric_path) if img.endswith(".npy")]
pred_depths_dav2_metric_npy = np.sort(pred_depths_dav2_metric_npy)

def eval_depth(gt, pred):

    scaling_factor = np.median(gt) / np.median(pred)
    pred = pred * scaling_factor

    abs_rel = np.mean(np.abs(pred - gt) / gt)
    sq_rel = np.mean(((pred - gt) ** 2) / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    thresh = np.maximum(pred / gt, gt / pred)
    d1 = np.mean(thresh < 1.25)


    return abs_rel, sq_rel, rmse, d1

for i in tqdm(range(len(pred_depths_dav2_metric_npy))):
    pred_depth_img = np.load(os.path.join(dav2_metric_path, pred_depths_dav2_metric_npy[i] + '.npy')) # for numpy version
    true_depth_img = cv2.imread(os.path.join(true_depth_path, true_depths[i]), cv2.IMREAD_UNCHANGED)  # Hamlyn - unit16
    true_depth_img = true_depth_img.astype(np.float32)

    valid_mask = (true_depth_img > 1)
    pred_valid = pred_depth_img[valid_mask]
    true_valid = true_depth_img[valid_mask]
    abs_rel, sq_rel, rmse, d1 = eval_depth(true_valid, pred_valid)

    abs_rel_all.append(abs_rel)
    sq_rel_all.append(sq_rel)
    rmse_all.append(rmse)
    d1_all.append(d1)

print('image01 results')
print({'abs_rel': np.mean(abs_rel_all), 'sq_rel': np.mean(sq_rel_all),'rmse': np.mean(rmse_all), 'd1': np.mean(d1_all)})

#### Second Image
abs_rel_02, sq_rel_02, rmse_02, d1_02 = [] , [] , [] , []
true_depth_path = r'/home/orifr/ori/datasets/Hamlyn_Dataset/rectified04/depth02'
pred_base_dir =  r'/home/orifr/ori/DLMI/Hamlyn_Dataset/rectified04/image02'
da_model = 'large'
dav1_path = os.path.join(pred_base_dir,'dav1', da_model)
dav2_path = os.path.join(pred_base_dir,'dav2', da_model)
dav2_metric_path = os.path.join(pred_base_dir,'dav2_metric', 'indoor_large')

true_depths = [img for img in os.listdir(true_depth_path) if img.endswith(".png")]
true_depths = np.sort(true_depths)

pred_depths_dav2_metric_npy =  [img[:-4] for img in os.listdir(dav2_metric_path) if img.endswith(".npy")]
pred_depths_dav2_metric_npy = np.sort(pred_depths_dav2_metric_npy)

for i in tqdm(range(len(pred_depths_dav2_metric_npy))):
    pred_depth_img = np.load(os.path.join(dav2_metric_path, pred_depths_dav2_metric_npy[i] + '.npy')) # for numpy version
    true_depth_img = cv2.imread(os.path.join(true_depth_path, true_depths[i]), cv2.IMREAD_UNCHANGED)  # Hamlyn - unit16
    true_depth_img = true_depth_img.astype(np.float32)

    valid_mask = (true_depth_img > 1)
    pred_valid = pred_depth_img[valid_mask]
    true_valid = true_depth_img[valid_mask]
    abs_rel, sq_rel, rmse, d1 = eval_depth(true_valid, pred_valid)

    abs_rel_all.append(abs_rel)
    sq_rel_all.append(sq_rel)
    rmse_all.append(rmse)
    d1_all.append(d1)

    abs_rel_02.append(abs_rel)
    sq_rel_02.append(sq_rel)
    rmse_02.append(rmse)
    d1_02.append(d1)

print('image02 results')
print({'abs_rel': "{:.4f}".format(np.mean(abs_rel_02)), 'sq_rel': "{:.4f}".format(np.mean(sq_rel_02)),'rmse': "{:.4f}".format(np.mean(rmse_02)), 'd1': "{:.4f}".format(np.mean(d1_02))})

print('all results')
print({'abs_rel': "{:.4f}".format(np.mean(abs_rel_all)), 'sq_rel': "{:.4f}".format(np.mean(sq_rel_all)),'rmse': "{:.4f}".format(np.mean(rmse_all)), 'd1': "{:.4f}".format(np.mean(d1_all))})