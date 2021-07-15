import numpy as np
import cv2 as cv
import os
import openpyxl as xl

def confusion_matrix(label, pred, n_class=2): #use np.bincount（）compute confusion matrix
    mask = (label >=0) & (label < n_class)
    hist = np.bincount(
        n_class*label[mask].astype(int) + pred[mask],
        minlength = n_class ** 2).reshape(n_class, n_class)
    return hist

def new_cm(label, pred):
    y_true = label.flatten()
    y_pred = pred.flatten()
    TP = np.sum(np.multiply(y_true, y_pred))
    TN = np.sum(np.multiply((1-y_true), (1-y_pred)))
    FP = np.sum(np.multiply((1-y_true), y_pred))
    FN = np.sum(np.multiply(y_true, (1-y_pred)))
    return TP, TN, FP, FN

def write_iou_xlsx(path, sheet_name, value): #save the results of MIoU to excel file
    index = len(value)
    workbook = xl.Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.cell(row=i+1, column=j+1, value=str(value[i][j]))
    workbook.save(path)
    print('successfully write the results to excel')


def iou(label_dir, pred_dir, model_name, xls_dir, type):
    n=0
    iou_crack=0
    iou_back=0
    total_iou_crack = 0
    total_iou_back = 0
    results = []
    results.append(['name', 'acc', 'pre', 'recall', 'F1', 'iou_c', 'iou_b', 'miou'])
    total_acc, total_pre, total_recall, total_F1 = 0, 0, 0, 0
    acc, pre, recall, F1 = 0, 0, 0, 0
    threshold=0.5
    for mask in os.listdir(label_dir):
        total_iou_crack += iou_crack
        total_iou_back += iou_back
        total_acc += acc
        total_pre += pre
        total_recall += recall
        total_F1 += F1
        label_file_path = os.path.join(label_dir, mask)
        label = cv.imread(label_file_path, 0)
        new_label = label/255
        label[new_label>threshold] = 0
        label[new_label<=threshold] = 1
        for test in os.listdir(pred_dir):
            if test == mask:
                pred_file_path = os.path.join(pred_dir, test)
                pred = (cv.imread(pred_file_path, 0)).astype(int)
                new_pred = pred/255
                pred[new_pred>threshold] = 1
                pred[new_pred<=threshold] = 0
                hist = confusion_matrix(label, pred) #compute confusion martix
                TP, TN, FP, FN = hist[1][1], hist[0][0], hist[0][1], hist[1][0]
                iou_crack = TP / (TP+FP+FN) #iou of single crack image =TP/(TP+FN+FP)
                iou_back = TN / (TN+FN+FP) #iou of single background image
                acc = (TP+TN)/(TP+FP+TN+FN)
                pre = TP/(TP+FP)
                recall = TP/(TP+FN)
                F1 = 2*(acc*recall)/(pre+recall)
                m_iou = (iou_crack + iou_back) / 2
                results.append([test.split('.')[0], str(acc), str(pre), str(recall), str(F1),
                                str(iou_crack), str(iou_back), str(m_iou),
                                str(TP), str(TN), str(FP), str(FN)]) #save the results of iou to the excel file
                print('img=', test)
        n=n+1
    m_iou_crack = total_iou_crack/n #Mean IoU of crack image in test set
    m_iou_back = total_iou_back/n #Mean IoU of background image in test set
    m_iou_total = (m_iou_crack+m_iou_back)/2 #mean iou of test set
    m_acc = total_acc/n
    m_pre = total_pre/n
    m_recall = total_recall/n
    m_F1 = total_F1/n
    results.append(['total',str(total_acc), str(total_pre), str(total_recall), str(total_F1),
                    str(m_iou_crack), str(m_iou_back), str(m_iou_total)])
    write_iou_xlsx(path=xls_dir+model_name+type+'.xlsx',
                   sheet_name=model_name, value=results)

