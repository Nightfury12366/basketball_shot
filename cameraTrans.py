import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from PIL import Image
from pylab import *
import pickle

# 4032*2268
# mtx:
#  [[3.27505277e+03, 0.00000000e+00, 2.03296344e+03],
#  [0.00000000e+00, 3.26117006e+03, 1.25836768e+03],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
# dist:
#  [[-6.51207000e-02,  1.91066376e+00,  4.87078007e-03,  4.06650667e-03, -1.37338065e+01]]

# cameraMatrix和distCoeffs是相机内参，可以由标定相机得到
camera_matrix = np.array(([[3.27505277e+03, 0.00000000e+00, 2.03296344e+03],
                           [0.00000000e+00, 3.26117006e+03, 1.25836768e+03],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), dtype=np.double)
dist_coefs = np.array([[-6.51207000e-02, 1.91066376e+00, 4.87078007e-03, 4.06650667e-03, -1.37338065e+01]],
                      dtype=np.double)  # 畸变参数

# 四个3d坐标点，坐标系参考我之前建立的
object_3d_points = np.array(([0., -6.750, 0.], [0., 6.750, 0.], [5.800, -1.800, 0.], [5.800, 1.800, 0.])
                            , dtype=np.double)
# 至少要4个点，一一对应，找到映射矩阵h
dst_point = np.float32([[30.0, 0.0], [570.0, 0.0], [228.0, 232.0], [372.0, 232.0]])  # 标准篮球半场四个对应点

dsize = (600, 560)  # 半场是15*14米, 所以虚拟场景设置为600*560的，所以一像素点表示1/40=0.025m
the_o = [300.0, 0.0]  # 定义篮球场地原点坐标
the_d = 0.025  # 一个像素格子是0.025米


# 求Zc
def calculator(perWidth):  # perWidth指的是当前像素宽度，篮球当时的像素宽度
    return (0.246 * 748.20822457) / perWidth


# 求Zc
def distance_to_camera(knownWidth, focalLength, perWidth):  # 另一个计算Zc的函数
    return (knownWidth * focalLength) / perWidth


# 重定义相机内参
def reFac_camera(sz):
    x_, y_ = sz
    camera_matrix[0, 0] = camera_matrix[0, 0] * x_ / 4032
    camera_matrix[1, 1] = camera_matrix[1, 1] * y_ / 2268
    camera_matrix[0, 2] = camera_matrix[0, 2] * x_ / 4032
    camera_matrix[1, 2] = camera_matrix[1, 2] * y_ / 2268


# 求二维两条直线的交点
def cross_point(line1, line2):  # 计算交点函数，直线由两个点表示
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 is None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


# 求平面
def get_panel(p1, p2, p3):  # 三个点确定一条直线，平面由4个参数组成
    a = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
    b = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
    c = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    d = 0 - (a * p1[0] + b * p1[1] + c * p1[2])
    panel = np.array([a, b, c, d], dtype=np.double)
    return panel


# 求直线与平面的交点
def cross_point_panel(p1, p2, panel):
    plane_normal = np.array([panel[0], panel[1], panel[2]])
    d = panel[3]
    P1D = (np.vdot(p1, plane_normal) + d) / np.sqrt(np.vdot(plane_normal, plane_normal))
    P1D2 = (np.vdot(p2 - p1, plane_normal)) / np.sqrt(np.vdot(plane_normal, plane_normal))
    n = abs(P1D / P1D2)
    # print("test_the_Panel:", n, P1D2, P1D, plane_normal, d)
    p = p1 + n * (p2 - p1)
    return p


#  从原始像素坐标变换到俯视图像素坐标
def cvt_pos(pos, cvt_mat_t):
    u = pos[0, 0]
    v = pos[0, 1]
    x = (cvt_mat_t[0][0] * u + cvt_mat_t[0][1] * v + cvt_mat_t[0][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
    y = (cvt_mat_t[1][0] * u + cvt_mat_t[1][1] * v + cvt_mat_t[1][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])

    return x, y


#  2D_to_3D (求射线终点)
def solve_2D_2_3D(point_2d, Pp_Matrix):
    p_ = np.array([point_2d[0], point_2d[1], 1], dtype=np.float)
    X_ = np.dot(Pp_Matrix, p_)
    X1_ = np.array(X_[:3], np.float) / X_[3]
    return X1_


# 求2D_to_3D的反变换矩阵以及相机位置
def solve_Pp_Matrix(object_2d_point):
    found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs)
    rotM = cv2.Rodrigues(rvec)[0]  # 生成旋转矩阵
    camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
    # 接下来尝试2D_to_3D
    Trans = np.hstack((rotM, tvec))
    Trans = np.vstack((Trans, np.array([0, 0, 0, 1])))
    print('Trans:', Trans)
    camera_matrix_ = np.hstack((camera_matrix, np.array([[0], [0], [0]])))
    temp = np.dot(camera_matrix_, Trans)
    Pp = np.linalg.pinv(temp)  # 求广义逆矩阵
    # print('旋转矩阵：', rotM, '\n', '平移向量：', tvec, '\n', '相机位置:', camera_postion.T)
    startPoint = np.array([camera_postion[0, 0], camera_postion[1, 0], camera_postion[2, 0]])  # 相机位置
    return Pp, startPoint


# 定义从俯视图像素坐标到世界俯视图坐标（m）的转换
def pixelXY_to_worldXY(pixelXY):
    print("像素坐标：", pixelXY)
    worldXY = pixelXY - np.array(the_o, dtype=np.double)
    print("世界俯视图坐标：", worldXY)
    worldXY = worldXY / 40.0
    return worldXY


# 绘制三维轨迹
def draw_3D_line(points_3ds):
    print("draw_3D_line")
    x1 = points_3ds[:, 0]  # [ 0  3  6  9 12 15 18 21]
    y1 = points_3ds[:, 1]  # [ 1  4  7 10 13 16 19 22]
    z1 = points_3ds[:, 2]  # [ 2  5  8 11 14 17 20 23]
    print(x1)
    print(y1)
    print(z1)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x1, y1, z1, c='r', label='篮球轨迹')
    # 绘制图例
    ax.legend(loc='best')
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # 展示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.show()


#  目前的主函数，我太难了
def sky_main():
    # 四个2d坐标点，开始标定工作
    im = array(Image.open('./image6/frame_1.jpg'))
    imshow(im)
    # 原图中的4个点
    src_point = ginput(4)
    # short_pointUV = ginput(1)
    # short_pointUV = np.array(short_pointUV, dtype=np.double)
    # print("short_pointUV: ", short_pointUV)
    object_2d_point = np.array(src_point, dtype=np.double)
    Pp, camera_postion = solve_Pp_Matrix(object_2d_point)  # 求解反推出来的2D_to_3D矩阵以及相机位置
    h, s = cv2.findHomography(object_2d_point, dst_point, cv2.RANSAC, 10)  # 求解单应性变换矩阵
    trans_photo = cv2.warpPerspective(im, h, dsize)  # 转俯视图
    trans_photo = cv2.cvtColor(trans_photo, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./image6/raw_tran.jpg', trans_photo)
    print("\n2D->3D矩阵：", Pp, '\n\n相机位置：', camera_postion, '\n')

    '''求Panel'''
    # short_pointXY = cvt_pos(short_pointUV, h)  # 求解出俯视图像素坐标
    # short_pointXY = pixelXY_to_worldXY(np.array(short_pointXY, dtype=np.double))  # 求解出俯视图世界坐标
    # shot_point = [short_pointXY[0], short_pointXY[1], 0]
    # print("shot_point:", shot_point)
    # hoop = np.array([1.575, 0.0, 3.05])
    # tripoint = np.array([1.575, 0.0, 0.0])
    # panel = get_panel(shot_point, hoop, tripoint)
    '''求Panel'''

    # point_3d_list = []
    # with open('shot_line.pkl', 'rb') as in_data:
    #     point_2d_list = pickle.load(in_data)
    #
    # print("篮球二维轨迹：", point_2d_list)
    #
    # for point_2d in point_2d_list:
    #     # point_2d = [474, 177]
    #     point_3d = solve_2D_2_3D(point_2d, Pp)
    #     the_cross_point = cross_point_panel(point_3d, camera_postion, panel)
    #     print("这个三维坐标点的估计(相交点估计)：", the_cross_point)
    #     point_3d_list.append(the_cross_point)
    #
    # point_3d_list = np.array(point_3d_list)
    # print("篮球三维轨迹：", point_3d_list)
    # draw_3D_line(point_3d_list)


reFac_camera((1280, 720))
print(camera_matrix)
sky_main()




# p1 = np.array([474, 177, 1], np.float)
# X = np.dot(Pp, p1)
#
# X1 = np.array(X[:3], np.float) / X[3]
# startPoint = np.array([camera_postion[0, 0], camera_postion[1, 0], camera_postion[2, 0]])  # 相机位置
#
# Face = X1 - startPoint
# N_Face = Face / sqrt(Face[0] * Face[0] + Face[1] * Face[1] + Face[2] * Face[2])
#
# Zc = calculator(14)  # 场景深度
# Z_len = sqrt(Zc * Zc + Face[2] * Face[2])  # 计算出的沿向量方向的位移
# X_3d = startPoint + Z_len * N_Face
#
# shot_point = [4.0, -4.0, 0]
# hoop = np.array([1.575, 0.0, 3.05])
# tripoint = np.array([1.575, 0.0, 0.0])
# panel = get_panel(shot_point, hoop, tripoint)
# print("这个平面是：", panel)
#
# the_cross_point = cross_point_panel(X1, startPoint, panel)
#
# print(startPoint, X1)
#
#
# print("这个三维坐标点的估计(深度估计)：", X_3d)
# print("这个三维坐标点的估计(相交点估计)：", the_cross_point)
