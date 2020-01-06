# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
import numpy as np
import cv2


# def cvt_pos(pos, cvt_mat_t):
#     u = pos[0]
#     v = pos[1]
#     x = (cvt_mat_t[0][0] * u + cvt_mat_t[0][1] * v + cvt_mat_t[0][2]) / (
#             cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
#     y = (cvt_mat_t[1][0] * u + cvt_mat_t[1][1] * v + cvt_mat_t[1][2]) / (
#             cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
#
#     return x, y


# if __name__ == '__main__':

# im = array(Image.open('./image6/raw.jpg'))
# imshow(im)
#
# point1 = np.array([158, 383], dtype=np.double)
#
# # 原图中的4个点
# src_point = ginput(4)
# src_point = np.float32(src_point)
#
# print(src_point)

# 想要图像的大小，（列数，行数）#先列数， 再行数
# dsize = (574, 546)
# im_ = array(Image.open('./image6/standard.jpg'))
# imshow(im_)
# dst_point = ginput(4)
# dst_point = np.float32(dst_point)
# print(dst_point)
# #


# dst_point = np.float32([[74.842476, 111.54859], [511.91144, 111.54859], [233.77664, 344.6011], [355.2696, 343.837]])
# # 至少要4个点，一一对应，找到映射矩阵h
# h, s = cv2.findHomography(src_point, dst_point, cv2.RANSAC, 10)
# book = cv2.warpPerspective(im, h, dsize)

# Image.open读入的图像是RGB，cv2.imwrite保存的是BGR


# *(xi, yi)
# 是原图像的坐标
#
# *(ui, vi)
# 是目标图像的坐标
#
# *
#
# *c00 * xi + c01 * yi + c02
#
# *ui = ---------------------
#
# *c20 * xi + c21 * yi + c22
#
# *
#
# *c10 * xi + c11 * yi + c12
#
# *vi = ---------------------
#
# *c20 * xi + c21 * yi + c22

# point1_ = cvt_pos(point1, h)
#
# print("point1_:", point1_)
# book = cv2.cvtColor(book, cv2.COLOR_RGB2BGR)
# cv2.imwrite('./image6/raw_tran.jpg', book)
# im_ = array(Image.open('./image6/raw_tran.jpg'))
# imshow(im_)
# plot(point1_[0], point1_[1], 'b*')
# show()
