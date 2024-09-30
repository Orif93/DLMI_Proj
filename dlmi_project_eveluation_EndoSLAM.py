import torch
import numpy as np
import os
import cv2
from tqdm import tqdm

abs_rel_all, sq_rel_all, rmse_all, d1_all = [] , [] , [] , []

true_depth_path = r'/home/orifr/ori/datasets/EndoSLAM_Dataset/UnityCam/Small_Intestine/Pixelwise Depths' #change between colon/stomach
true_depths = [img for img in os.listdir(true_depth_path) if img.endswith(".png")]
true_depths = np.sort(true_depths)

dav2_path = r'/home/orifr/ori/DLMI/EndoSLAM_Dataset/UnityCam/Small_Intestine/dav2' #change between dav1 and dav2
pred_depths_dav2 =  [img for img in os.listdir(dav2_path) if img.endswith(".png")]
pred_depths_dav2 = np.sort(pred_depths_dav2)

def eval_depth(gt, pred):

    scaling_factor = np.median(gt) / np.median(pred)
    pred = pred * scaling_factor

    abs_rel = np.nanmean(np.abs(pred - gt) / gt)
    sq_rel = np.nanmean(((pred - gt) ** 2) / gt)


    rmse = np.sqrt(np.nanmean((pred - gt) ** 2))
    thresh = np.maximum(pred / gt, gt / pred)
    d1 = np.nanmean(thresh < 1.25)


    return abs_rel, sq_rel, rmse, d1

def eval_depth_no_masking(gt, pred):

    scaling_factor = np.median(gt) / np.median(pred)
    pred = pred * scaling_factor

    abs_rel = np.nanmean((np.abs(pred - gt) / gt)[np.isfinite(np.abs(pred - gt) / gt)]) #No Masking
    sq_rel_item = ((pred - gt) ** 2) / gt #No Masking
    sq_rel = np.nanmean(sq_rel_item[np.isfinite(sq_rel_item)]) #No Masking

    rmse = np.sqrt(np.nanmean((pred - gt) ** 2))
    thresh = np.maximum(pred / gt, gt / pred)
    d1 = np.nanmean(thresh < 1.25)


    return abs_rel, sq_rel, rmse, d1

for i in tqdm(range(len(pred_depths_dav2))):
    pred_depth_img = cv2.imread(os.path.join(dav2_path, pred_depths_dav2[i]), cv2.IMREAD_UNCHANGED)
    true_depth_img = cv2.imread(os.path.join(true_depth_path, true_depths[i]), cv2.IMREAD_UNCHANGED)  # EndoSLALM - unit8 (320,320,4)   true_depth_img = cv2.cvtColor(true_depth_img, cv2.COLOR_BGR2GRAY) # only for EndoSLAM
    true_depth_img = true_depth_img[:,:,:3]

    valid_mask = (true_depth_img > 0) & (pred_depth_img > 0)
    pred_valid = pred_depth_img[valid_mask]
    true_valid = true_depth_img[valid_mask]

    abs_rel, sq_rel, rmse, d1 = eval_depth_no_masking(true_depth_img, pred_depth_img)

    abs_rel_all.append(abs_rel)
    sq_rel_all.append(sq_rel)
    rmse_all.append(rmse)
    d1_all.append(d1)

print({'abs_rel': "{:.4f}".format(np.nanmean(abs_rel_all)) ,'rmse': "{:.4f}".format(np.nanmean(rmse_all)), 'sq_rel': "{:.4f}".format(np.nanmean(sq_rel_all)), 'd1': "{:.4f}".format(np.nanmean(d1_all))})