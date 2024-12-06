import numpy as np
from fitness import fitness
import random
from InitialPops import find_min_3d_array
from tkinter import _flatten


def fusion(pops,  q, w, input_x_data, input_y_data):
    N = len(pops)
    Mates = np.random.choice(range(N), (w, N), replace=True)
    # 两个父代融合，生成寄存个体集regMates
    fuspops = []
    for col in range(N):  # 遍历list1的每一列
        row_indices = set(Mates[row][col] for row in range(w))  # 获取该列对应的所有行的索引并去重
        if len(row_indices) == 1:
            fuspops.append(pops[list(row_indices)[0]])
        else:
            merged_row = []
            for index in row_indices:
                t_pop = list(_flatten(pops[index]))
                t_pop = [idx for idx in t_pop if idx != -1]
                merged_row.append(t_pop)
            merged_row = list(set(list(_flatten(merged_row))))
            fuspops.append(find_min_3d_array(merged_row))
    fusFits = fitness(fuspops, input_x_data, input_y_data)
    fusq = []
    for j in range(N):
        max_val = float('-inf')
        for i in range(w):
            max_val = max(max_val, q[Mates[i][j]])
        fusq.append(max_val)
    return fuspops, fusFits, fusq
