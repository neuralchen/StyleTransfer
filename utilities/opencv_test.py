#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: opencv_test.py
# Created Date: Thursday April 23rd 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 23rd April 2020 11:52:19 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import cv2


def mouse(event, x, y, flags, param):
    global flag, x1, y1, x2, y2, wx, wy, move_w, move_h, dst
    global zoom, zoom_w, zoom_h, img_zoom, flag_har, flag_var
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        if flag == 0:
            flag = 1
            x1, y1, x2, y2 = x, y, wx, wy  # 使鼠标移动距离都是相对于初始点击位置，而不是相对于上一位置
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        if flag == 1:
            move_w, move_h = x1 - x, y1 - y  # 鼠标拖拽移动的宽高
            if flag_har and flag_var:  # 当窗口宽高大于图片宽高
                wx = x2 + move_w  # 窗口在大图的横坐标
                if wx < 0:  # 矫正位置
                    wx = 0
                elif wx + win_w > zoom_w:
                    wx = zoom_w - win_w
                wy = y2 + move_h  # 窗口在大图的总坐标
                if wy < 0:
                    wy = 0
                elif wy + win_h > zoom_h:
                    wy = zoom_h - win_h
                dst = img_zoom[wy:wy + win_h, wx: wx + win_w]  # 截取窗口显示区域
            elif flag_har and flag_var == 0:  # 当窗口宽度大于图片宽度
                wx = x2 + move_w
                if wx < 0:
                    wx = 0
                elif wx + win_w > zoom_w:
                    wx = zoom_w - win_w
                dst = img_zoom[0:zoom_h, wx: wx + win_w]
            elif flag_har == 0 and flag_var:  # 当窗口高度大于图片高度
                wy = y2 + move_h
                if wy < 0:
                    wy = 0
                elif wy + win_h > zoom_h:
                    wy = zoom_h - win_h
                dst = img_zoom[wy:wy + win_h, 0: zoom_w]
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        flag = 0
        x1, y1, x2, y2 = 0, 0, 0, 0
    elif event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
        z = zoom
        if flags > 0:  # 滚轮上移
            zoom += wheel_step
            if zoom > 1 + wheel_step * 20:  # 缩放倍数调整
                zoom = 1 + wheel_step * 20
        else:  # 滚轮下移
            zoom -= wheel_step
            if zoom < wheel_step:  # 缩放倍数调整
                zoom = wheel_step
        zoom = round(zoom, 2)  # 取2位有效数字
        zoom_w, zoom_h = int(img_original_w * zoom), int(img_original_h * zoom)
        print(wx, wy)
        wx, wy = int((wx + x) * zoom / z - x), int((wy + y) * zoom / z - y)  # 缩放后鼠标在原图的坐标
        # print(z, zoom, x, y, wx, wy)
        if wx < 0:
            wx = 0
        elif wx + win_w > zoom_w:
            wx = zoom_w - win_w
        if wy < 0:
            wy = 0
        elif wy + win_h > zoom_h:
            wy = zoom_h - win_h
        img_zoom = cv2.resize(img_original, (zoom_w, zoom_h), interpolation=cv2.INTER_AREA)  # 图片缩放
        if zoom_w <= win_w and zoom_h <= win_h:  # 缩放后图片宽高小于窗口宽高
            flag_har, flag_var = 0, 0
            dst = img_zoom
            cv2.resizeWindow('img', zoom_w, zoom_h)
        elif zoom_w <= win_w and zoom_h > win_h:  # 缩放后图片宽度小于窗口宽度
            flag_har, flag_var = 0, 1
            dst = img_zoom[wy:wy + win_h, 0:zoom_w]
            cv2.resizeWindow('img', zoom_w, win_h)
        elif zoom_w > win_w and zoom_h <= win_h:  # 缩放后图片高度小于窗口高度
            flag_har, flag_var = 1, 0
            dst = img_zoom[0:zoom_h, wx:wx + win_w]
            cv2.resizeWindow('img', win_w, zoom_h)
        else:  # 缩放后图片宽高大于于窗口宽高
            flag_har, flag_var = 1, 1
            dst = img_zoom[wy:wy + win_h, wx:wx + win_w]
            cv2.resizeWindow('img', win_w, win_h)
    cv2.imshow("img", dst)
    cv2.waitKey(1)


img_original = cv2.imread("D:\\PatchFace\\PleaseWork\\multi-style-gan\\StyleTransfer\\test_logs\\SN-FC512_ms_4\\samples\\6.jpg_step78000_v_SN-FC512_ms_4.png")  # 此处需换成大于img_w * img_h的图片
img_original_h, img_original_w = img_original.shape[0:2]  # 原图宽高
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.moveWindow("img", 300, 100)
win_h, win_w = 600, 800  # 窗口宽高
wx, wy = 0, 0  # 窗口相对于原图的坐标
wheel_step, zoom = 0.05, 1  # 缩放系数， 缩放值
zoom_w, zoom_h = img_original_w, img_original_h  # 缩放图宽高
img_zoom = img_original.copy()  # 缩放图片
flag, flag_har, flag_var = 0, 0, 0  # 鼠标操作类型
move_w, move_h = 0, 0  # 鼠标移动坐标
x1, y1, x2, y2 = 0, 0, 0, 0  # 中间变量
cv2.resizeWindow("img", win_w, win_h)
dst = img_original[wy:wy + win_h, wx: wx + win_w]
cv2.setMouseCallback('img', mouse)
if img_original_w > win_w:
    flag_har = 1
if img_original_h > win_h:
    flag_var = 1
cv2.waitKey()
cv2.destroyAllWindows()