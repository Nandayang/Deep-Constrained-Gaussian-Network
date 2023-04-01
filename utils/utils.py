import numpy as np
import tensorflow as tf
import cv2
from scipy import ndimage as ndi
from tqdm import tqdm
import numexpr as ne

def dataset_normalized(img):
    assert len(img.shape)==3
    img_normalized = np.empty(img.shape)
    for i in range(img.shape[2]):
        img_temp = img[:,:,i]
        img_temp_std = np.std(img_temp)
        img_temp_mean = np.mean(img_temp)
        img_temp_normalized = (img_temp - img_temp_mean) / img_temp_std
        img_normalized[:,:,i] = ((img_temp_normalized - np.min(img_temp_normalized))/
                                 (np.max(img_temp_normalized)- np.min(img_temp_normalized)))*255
    return img_normalized


def compute_dice(pred, gt, smooth=1):
    pred_f = np.reshape(np.round(pred), (-1))
    gt_f = np.reshape(gt, (-1))
    if np.max(gt_f)==0:
        pred_f  = 1- pred_f
        gt_f = 1 - gt_f
    I = 2*(np.sum(gt_f*pred_f))
    U = (np.sum(gt_f) + np.sum(pred_f))
    mean_dice = np.mean((I+smooth) / (U+smooth))
    return mean_dice


def get_bn_vars(collection):
    moving_mean, moving_variance = None, None
    for var in collection:
        name = var.name.lower()
        if "variance" in name:
            moving_variance = var
        if "mean" in name:
            moving_mean = var

    if moving_mean is not None and moving_variance is not None:
        return moving_mean, moving_variance
    raise ValueError("Unable to find moving mean and variance")


def tfdice_loss(pred, gt):
    smooth = 1.0
    pred_f = tf.squeeze(pred)
    gt_f = tf.squeeze(gt)
    if tf.reduce_max(gt_f) == 0:
        pred_f = 1 - pred_f
        gt_f = 1 - gt_f
    intersection = 2.0 *tf.reduce_sum(pred_f * gt_f)
    U = tf.reduce_sum(gt_f) + tf.reduce_sum(pred_f)
    bce = tf.keras.losses.binary_crossentropy(gt_f, pred_f)
    return 1 - intersection/U


def FillHole(mask):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, 1, -1)
        contour_list.append(img_contour)
    out = sum(contour_list)
    return out


def color_normalization(srcImg):
    grayImg = cv2.cvtColor(srcImg, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(grayImg, 230, 255, cv2.THRESH_BINARY_INV)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.dilate(mask, erode_kernel)
    meanTargetBGR = [181.1, 142.6,196.65]
    stdTargetBGR = [28.28,41.21,27.14]

    srcImg = srcImg.astype(np.uint8)
    srcB,srcG,srcR = cv2.split(srcImg)
    meanSrcB, stdSrcB = cv2.meanStdDev(srcB, mask=mask)
    meanSrcG, stdSrcG = cv2.meanStdDev(srcG, mask=mask)
    meanSrcR, stdSrcR = cv2.meanStdDev(srcR, mask=mask)

    arrNormBGR = np.zeros((srcImg.shape[0], srcImg.shape[1],3))
    arrNormBGR[:,:,0] = (srcB-meanSrcB)/stdSrcB*stdTargetBGR[0]+meanTargetBGR[0]
    arrNormBGR[:,:,1] = (srcG-meanSrcG)/stdSrcG*stdTargetBGR[1]+meanTargetBGR[1]
    arrNormBGR[:,:,2] = (srcR-meanSrcR)/stdSrcR*stdTargetBGR[2]+meanTargetBGR[2]

    arrNormBGR[arrNormBGR>255]=255
    arrNormBGR[arrNormBGR<0] = 0
    arrNormBGR[mask==0] = srcImg[mask==0]
    arrNormBGR = arrNormBGR.astype(np.uint8)
    return arrNormBGR

import matplotlib.pyplot as plt
def agg_jc_index(mask, pred):
    from tqdm import tqdm_notebook
    """Calculate aggregated jaccard index for prediction & GT mask
    reference paper here: https://www.dropbox.com/s/j3154xgkkpkri9w/IEEE_TMI_NuceliSegmentation.pdf?dl=0
    mask: Ground truth mask, shape = [1000, 1000, instances]
    pred: Prediction mask, shape = [1000,1000], dtype = uint16, each number represent one instance
    Returns: Aggregated Jaccard index for GT & mask 
    """
    pred_ori = np.round(cv2.resize(pred, (1000,1000)))
    _, contours, _ = cv2.findContours(pred_ori.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_refine = np.zeros_like(pred_ori,np.uint16)
    for i in range(len(contours)+1):
        if cv2.contourArea(contours[i-1])<40:
            continue
        cv2.fillPoly(img=pred_refine, pts=[contours[i-1]], color=(i+1))
    pred = pred_refine
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(pred_refine)
    # plt.subplot(122)
    # plt.imshow(np.sum(mask,-1))
    # plt.show()

    def compute_iou(m, pred, pred_mark_isused, idx_pred):
        # check the prediction has been used or not
        if pred_mark_isused[idx_pred]:
            intersect = 0
            union = np.count_nonzero(m)
        else:
            p = (pred == idx_pred)
            # replace multiply with bool operation
            s = ne.evaluate("m&p")
            intersect = np.count_nonzero(s)
            union = np.count_nonzero(m) + np.count_nonzero(p) - intersect
        return (intersect, union)

    mask = mask.astype(np.bool)
    c = 0  # count intersection
    u = 0  # count union
    pred_instance = pred.max()  # predcition instance number
    pred_mark_used = []  # mask used
    pred_mark_isused = np.zeros((pred_instance + 1), dtype=bool)

    for idx_m in range(len(mask[0, 0, :])):
        m = np.take(mask, idx_m, axis=2)

        intersect_list, union_list = zip(
            *[compute_iou(m, pred, pred_mark_isused, idx_pred) for idx_pred in range(1, pred_instance + 1)])

        iou_list = np.array(intersect_list) / np.array(union_list)
        hit_idx = np.argmax(iou_list)
        c += intersect_list[hit_idx]
        u += union_list[hit_idx]
        pred_mark_used.append(hit_idx)
        pred_mark_isused[hit_idx + 1] = True

    pred_mark_used = [x + 1 for x in pred_mark_used]
    pred_fp = set(np.unique(pred)) - {0} - set(pred_mark_used)
    pred_fp_pixel = np.sum([np.sum(pred == i) for i in pred_fp])

    u += pred_fp_pixel
    print(c / u)
    return (c / u)

def IOU(pred, gt, smooth=1):
    pred_f = np.round(np.reshape(pred, -1))
    gt_f = np.reshape(gt, (-1))

    I = np.sum(gt_f*pred_f)
    U = np.sum(gt_f) + np.sum(pred_f)-I
    return (I+smooth)/(U+smooth)
