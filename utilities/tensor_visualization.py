#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tensor_visualization.py
# Created Date: Thursday April 23rd 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 8th May 2020 10:50:31 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


# coding: utf-8
import cv2
import numpy as np
img = cv2.imread("D:\\PatchFace\\PleaseWork\\multi-style-gan\\StyleTransfer\\test_logs\\SN-FC512_ms_4\\samples\\6.jpg_step278000_v_SN-FC512_ms_4.png")
img_tensor = np.load("D:\\PatchFace\\PleaseWork\\multi-style-gan\\StyleTransfer\\test_logs\\SN-FC512_ms_4\\samples\\6.jpg_step278000_v_SN-FC512_ms_4.npz")
img_tensor = img_tensor['arr_0']

# print img.shape
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "[%d,%d]-value[%f,%f,%f]"%(x,y,img_tensor[0,y,x],img_tensor[1,y,x],img_tensor[2,y,x])
        wocao = img.copy()
        cv2.putText(wocao, xy, (0, 20), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 0), thickness=2)
        cv2.imshow("image", wocao)


cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyAllWindows()
        break

cv2.waitKey(0)
cv2.destroyAllWindows()