import cv2 as cv
import numpy as np
from Cal_IoU import new_cm as cm
from Cal_IoU import write_iou_xlsx as write
import os
def mask2bi(img):
    img = np.array(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thre, bi = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
    return bi

def cal_resize_iou(ori_mask_path, pred_mask_path, write_xlsx_path, model_name):
    results = []
    results.append(['name', 'acc', 'pre', 'recall', 'F1', 'iou_c', 'iou_b', 'miou', 'TP', 'TN', 'FP', 'FN'])
    total_acc, total_pre, total_recall, total_F1 = 0, 0, 0, 0
    acc, pre, recall, F1 = 0, 0, 0, 0
    n = 0
    threshold=0.5
    total_iou_crack = 0
    total_iou_back = 0
    for img in os.listdir(pred_mask_path):
                # step 1: read an original label and convert it into bi-mask
        ori_mask = cv.imread(ori_mask_path+img.split('.')[0]+'.png')
        ori_mask = mask2bi(ori_mask)
        new_ori_mask = ori_mask/255
        ori_mask[new_ori_mask>threshold] = 1
        ori_mask[new_ori_mask<=threshold] = 0

            # step 2: read a predicted label and convert it into bi-mask
        pred_mask = cv.imread(pred_mask_path+img, -1)
        new_pred_mask = pred_mask/255
        pred_mask[new_pred_mask>threshold] = 1
        pred_mask[new_pred_mask<=threshold] = 0
        pred_resize = cv.resize(pred_mask, (ori_mask.shape[1], ori_mask.shape[0]))
        TP, TN, FP, FN = cm(ori_mask, pred_resize)
        iou_crack = TP/(TP+FP+FN)
        iou_background = TN/(TN+FN+FP)
        acc = (TP + TN) / (TP + FP + TN + FN)
        pre = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (acc * recall) / (pre + recall)
        miou = (iou_crack+iou_background)/2
        results.append([img.split('.')[0], str(acc), str(pre), str(recall), str(F1),
                        str(iou_crack), str(iou_background), str(miou), str(TP), str(TN), str(FP), str(FN)])
        print('resize_img=', img)
        n += 1
        total_iou_crack += iou_crack
        total_iou_back += iou_background
        total_acc += acc
        total_pre += pre
        total_recall += recall
        total_F1 += F1
    miou_crack = total_iou_crack/n
    miou_back = total_iou_back/n
    miou_total = (miou_crack+miou_back)/2
    m_acc = total_acc/n
    m_pre = total_pre/n
    m_recall = total_recall/n
    m_F1 = total_F1/n
    results.append(['total',str(total_acc), str(total_pre), str(total_recall), str(total_F1),
                   str(miou_crack), str(miou_back), str(miou_total)])
    write(path=write_xlsx_path+ model_name+'_resize.xlsx', sheet_name=model_name+'_resize',
          value=results)





