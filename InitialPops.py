import numpy as np
import warnings
import math

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

"""
初始化群体
"""


def initpops(nPop, nFeature):
    """
    :param nPop: 种群规模大小
    :param nFeature: 特征总数
    :return:
    """
    pops = []
    # 随机群体
    for i in range(nPop):
        index = range(nFeature)  # 产生特征索引值
        # nChar = np.random.randint(low=1, high=10)  # 随机生成一个[0,nFeature)的整数作为个体的长度
        pop = list(np.random.choice(index, nFeature//10, replace=False))  # 从特征索引列表选取n_char个索引生成个体
        pops.append(find_min_3d_array(pop))
    return pops


def find_min_3d_array(lst):
    """
    转换为3维立体编码
    :param lst:
    :return:
    """
    n = len(lst)
    shape_list = []

    # 遍历可能的 x, y, z
    for x in range(1, int(math.ceil(n ** (1 / 3))) + 1):  # x 尽量小但 >= 立方根
        for y in range(2, int(math.ceil(n / x ** 0.5)) + 1):  # y >= x
            z = math.ceil(n / (x * y))
            volume = x * y * z
            if volume >= n:
                shape_list.append([x, y, int(z)])

    # 希望矩阵尺寸尽可能匀称，同时不要出现一维长度为1的情况
    sum_s_list = [np.sum(i) for i in shape_list]
    filtrated_shape_index_list = [i for i in range(len(shape_list)) if sum_s_list[i] == np.min(sum_s_list)]
    x, y, z = shape_list[np.random.choice(filtrated_shape_index_list, 1)[0]]

    result = [[[-1 for _ in range(z)] for _ in range(y)] for _ in range(x)]
    # 填充三维数组
    idx = 0
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if idx < n:
                    result[i][j][k] = lst[idx]
                    idx += 1
    return result

