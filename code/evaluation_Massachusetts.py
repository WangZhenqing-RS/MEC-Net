# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:16:58 2022

@author: DELL
"""
import numpy as np
import cv2
import os
import glob
from scipy.ndimage.morphology import distance_transform_edt

def get_edge_from_binary_image(binary_image, radius=2):
    
    binary_image_pad = np.pad(binary_image, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    dist = distance_transform_edt(binary_image_pad) + distance_transform_edt(1.0 - binary_image_pad)
    dist = dist[1:-1, 1:-1]
    dist[dist > radius] = 0
    edge = (dist > 0).astype(np.uint8)
    
    edge[:2, :] = 0
    edge[-2:, :] = 0
    edge[:, :2] = 0
    edge[:, -2:] = 0
    
    return edge


""" 
混淆矩阵
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""  

def ConfusionMatrix(numClass, imgPredict, Label):  
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix

def OverallAccuracy(confusionMatrix):  
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)  
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA
  
def Precision(confusionMatrix):  
    #  返回所有类别的精确率precision  
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    return precision  

def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return recall
  
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score
def IntersectionOverUnion(confusionMatrix):  
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):  
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def evaluation(label_folder, pred_floder, only_center=False):
    
    classNum = 2
    label_paths = glob.glob(label_folder)
    pred_paths = glob.glob(pred_floder)
    
    label_all = []
    predict_all = []
    label_egde_area_all = []
    predict_edge_area_all = []
    
    for i in range(len(label_paths)):
        label = cv2.imread(label_paths[i],0)
        label[label!=0] = 1
        if only_center:
            label = label[128:384,128:384]
        
        label_egde = get_edge_from_binary_image(label)
        label_egde_area = label.copy()
        label_egde_area[label_egde==0] = 2
        
        label_all.append(label)
        label_egde_area_all.append(label_egde_area)
        
        
        predict = cv2.imread(pred_paths[i],0)
        predict[predict!=0] = 1
        if only_center:
            predict = predict[128:384,128:384]
            
        predict_edge_area = predict.copy()
        predict_edge_area[label_egde==0] = 2
        
        predict_all.append(predict)
        predict_edge_area_all.append(predict_edge_area)
    
    label_all = np.array(label_all).flatten()
    predict_all = np.array(predict_all).flatten()
    label_egde_area_all = np.array(label_egde_area_all).flatten()
    predict_edge_area_all = np.array(predict_edge_area_all).flatten()
    
    
    #  计算混淆矩阵及各精度参数
    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    # OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    # FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    # mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)
    
    label_egde_area_all = np.delete(label_egde_area_all,
                                    np.where(label_egde_area_all==2)[0])
    predict_edge_area_all = np.delete(predict_edge_area_all,
                                      np.where(predict_edge_area_all==2)[0])
    confusionMatrix_edge_area = ConfusionMatrix(classNum, predict_edge_area_all, label_egde_area_all)
    IoU_edge_area = IntersectionOverUnion(confusionMatrix_edge_area)
    
    
    print(f"precision: {round(precision[1]*100,2)}")
    print(f"recall: {round(recall[1]*100,2)}")
    print(f"F1-Score: {round(f1ccore[1]*100,2)}")
    print(f"IoU: {round(IoU[1]*100,2)}")
    print(f"IoU_edge: {round(IoU_edge_area[1]*100,2)}")

class decoder_names:
    DeepLabV3Plus = "DeepLabV3Plus"
    ResUNet = "ResU-Net"
    PSPNet = "PSP-Net"
    HRNet = "HR-Net"
    MECNET = "MEC-Net"
    UentPlusPlus = "Unet++"
    
if __name__ == "__main__":
    
    # decoder_name_list = [
    #     decoder_names.PSPNet,
    #     decoder_names.ResUNet,
    #     decoder_names.HRNet,
    #     decoder_names.DeepLabV3Plus,
    #     decoder_names.UentPlusPlus,
    #     decoder_names.MECNET]
    
    # for decoder_name in decoder_name_list:
    #     label_folder = '../data/Massachusetts_Building_Dataset/test/label/*.tif'
    #     pred_floder = f'../data/Massachusetts_Building_Dataset/test/pred_{decoder_name}/*.tif'
    #     print("====================================")
    #     print(decoder_name)
    #     print("============================")
    #     evaluation(label_folder, pred_floder)
        
    # label_folder = '../data/WHU_Building_Dataset/test/label/*.tif'
    # pred_floder = f'../data/WHU_Building_Dataset/test/pred_{decoder_names.MECNET}_build_building/*.tif'
    # print("====================================")
    # print(f"{decoder_names.MECNET}_build_building")
    # print("================================")
    # evaluation(label_folder, pred_floder)
    
    label_folder = '../data/Massachusetts_Building_Dataset/test/label/*.tif'
    pred_floder = '../data/Massachusetts_Building_Dataset/test/pred_fusion/*.tif'
    print("====================================")
    print("fusion")
    print("================================")
    evaluation(label_folder, pred_floder, only_center=False)