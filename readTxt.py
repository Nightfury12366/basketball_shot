# -*- coding: utf-8 -*-
import numpy as np

F1 = open(r"C:\Users\64426\Desktop\ball_line\3 00_00_38-00_00_41.txt", "r")

List_row = F1.readlines()

list_source = []
for i in range(len(List_row)):
    column_list = List_row[i].strip().split(",")  # 每一行split后是一个列表
    list_source.append(column_list)  # 加入list_source

# for i in range(len(list_source)):  # 行数
#     for j in range(len(list_source[i])):  # 列数
#         print(list_source[i][j])  # 输出每一项
list_source = np.array(list_source, dtype=np.float)

print(list_source)


