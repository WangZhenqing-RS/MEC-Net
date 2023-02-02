# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:05:07 2022

@author: DELL
"""
import cv2
import glob
import numpy as np
import tqdm

def cal_ratio(label_paths):
    p_n_num = np.zeros((2),np.float64)
    for label_path in tqdm.tqdm(label_paths):
        label = cv2.imread(label_path,0)
        p_n_num[0] += np.sum(label!=0)
        p_n_num[1] += np.sum(label==0)
    print(f"p_num: {p_n_num[0]}, n_num: {p_n_num[1]}, p/n: {round(p_n_num[0]/p_n_num[1],2)}")
    
WHU_label_paths = glob.glob('../data/WHU_Building_Dataset/train/label/*.tif')
cal_ratio(WHU_label_paths) # 0.23
Massachusetts_label_paths = glob.glob('../data/Massachusetts_Building_Dataset/train/label/*.tif')
cal_ratio(Massachusetts_label_paths) # 0.15
Inria_label_paths = glob.glob('../data/Inria_Building_Dataset/train/label/*.tif')
cal_ratio(Inria_label_paths) # 0.24