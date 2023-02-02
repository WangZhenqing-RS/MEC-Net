# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:37:34 2021

@author: DELL
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def grade(img):
    x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    mi=np.min(dst)
    ma=np.max(dst)
    res=(dst-mi)/(0.000000001+(ma-mi))
    res[np.isnan(res)]=0
    return res


def plt_grad(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad = grade(gray)
    # grad = (255 * grade(gray)).astype(np.uint8)
    
    plt.imshow(grad)
    plt.colorbar(fraction=0.05)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(image_path.replace(".tif","_grad.tif"),dpi=600)
    plt.clf()
    # cv2.imwrite(r"E:\WangZhenQing\2021GaoFen\code\41_1_grad.png",grad)

image_path = "WHU.tif"
plt_grad(image_path)
image_path = "Massachusetts.tif"
plt_grad(image_path)
image_path = "Inria.tif"
plt_grad(image_path)