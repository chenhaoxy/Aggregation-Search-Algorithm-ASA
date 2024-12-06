import numpy as np
import random
from InitialPops import find_min_3d_array
from function import func
from tkinter import _flatten

"""   
分裂 fission
涉及参数：长度小于3的个体保留概率rate
1 先根据分裂概率进行分裂
2 分裂之后计算△q
3 更新分裂概率
"""


def fission(fuspops, fusFits, fusq, input_x_data, input_y_data):
    # 定义本次迭代分裂操作之后获得的新一代种群、适应度值、收益值
    fispops = []
    fisFits = np.array([[0, 0]])

    """第一种分裂"""
    # for i, pop in enumerate(fuspops):
    #     if np.random.random() < fusq[i]:
    #         f_pop = list(_flatten(pop))
    #         f_pop = [idx for idx in f_pop if idx != -1]
    #         np.random.shuffle(f_pop)
    #         # 随机决定分裂成多少个子合团
    #         num_splits = np.random.randint(2, 5)  # 允许分裂成多个合团
    #         split_features_list = np.array_split(f_pop, num_splits)
    #         for split_features in split_features_list:
    #             # 转换为三维矩阵
    #             new_group = find_min_3d_array(split_features)
    #             fispops.append(new_group)
    #             pop_fit = np.array(func(split_features, input_x_data, input_y_data))
    #             fisFits = np.append(fisFits, pop_fit.reshape(1, 2), 0)
    #     else:
    #         fispops.append(pop)
    #         fisFits = np.append(fisFits, fusFits[i].reshape(1, 2), 0)

    """第二种分裂"""
    for i, pop in enumerate(fuspops):
        pop = np.array(pop)
        if np.random.random() < fusq[i]:
            x, y, z = pop.shape
            cut_x, cut_y, cut_z = 0, 0, 0
            if x > 1:
                cut_x = random.randint(1, x-1)
            if y > 1:
                cut_y = random.randint(1, y-1)
            if z > 1:
                cut_z = random.randint(1, z-1)

            part_1, part_2 = pop[0:cut_x, 0:cut_y, 0:cut_z].flatten(), pop[0:cut_x, 0:cut_y, cut_z:z].flatten()
            part_3, part_4 = pop[0:cut_x, cut_y:y, cut_z:z].flatten(), pop[0:cut_x, cut_y:y, 0:cut_z].flatten()
            part_5, part_6 = pop[cut_x:x, 0:cut_y, 0:cut_z].flatten(), pop[cut_x:x, 0:cut_y, cut_z:z].flatten()
            part_7, part_8 = pop[cut_x:x, cut_y:y, cut_z:z].flatten(), pop[cut_x:x, cut_y:y, 0:cut_z].flatten()
            part_list = [part_1, part_2, part_3, part_4, part_5, part_6, part_7, part_8]
            first_id_list = random.sample(range(8), 4)
            second_id_list = [i for i in range(8) if i not in first_id_list]
            first_ind = list(set(_flatten([list(part_list[i]) for i in first_id_list])))
            second_ind = list(set(_flatten([list(part_list[i]) for i in second_id_list])))
            fispops.append(find_min_3d_array(first_ind))
            fispops.append(find_min_3d_array(second_ind))
            fisFits = np.append(fisFits, np.array(func(first_ind, input_x_data, input_y_data)).reshape(1, 2), 0)
            fisFits = np.append(fisFits, np.array(func(second_ind, input_x_data, input_y_data)).reshape(1, 2), 0)

    # 分裂结束后， 去掉fisFits第一个空的（0，0）
    fisFits = np.delete(fisFits, 0, 0)
    return fispops, fisFits
