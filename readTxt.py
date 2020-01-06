# -*- coding: utf-8 -*-
import numpy as np

F1 = open(r"C:\Users\64426\Desktop\ball_line\3 00_00_38-00_00_41.txt", "r")

List_row = F1.readlines()

list_source = []
list_target = []

for i in range(len(List_row)):
    column_list = List_row[i].strip().split(",")  # 每一行split后是一个列表
    list_source.append(column_list)  # 加入list_source

# for i in range(len(list_source)):  # 行数
#     for j in range(len(list_source[i])):  # 列数
#         print(list_source[i][j])  # 输出每一项
list_source = np.array(list_source, dtype=np.float)

for row in list_source:
    row_target = np.array([(row[0] + row[2]) / 2.0, (row[1] + row[3]) / 2.0], dtype=np.float)
    if row_target.sum() != 0:
        list_target.append(row_target)

list_target = np.array(list_target, dtype=np.float)

print(list_target)
