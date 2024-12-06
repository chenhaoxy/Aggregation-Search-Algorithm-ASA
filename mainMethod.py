import numpy as np
from fitness import fitness
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import warnings
from fissionFun import fission
from fusionFun import fusion
import matplotlib.pyplot as plt
from InitialPops import initpops
from updateFun import update
from bestFun import best
import os
from tkinter import _flatten

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)


def faf(N, L, T, qvalue, w, input_x_data, input_y_data):
    """
    :param N: 种群规模
    :param L: 特征总数
    :param T: 最大迭代次数
    :param qvalue: 初始分裂概率
    :param w: 一次选取合团数量
    :return:
    """
    # step1 初始化
    pops = initpops(N, L)
    fits = fitness(pops, input_x_data, input_y_data)
    q = np.full(N, qvalue, dtype=float)
    best_individual, bestAcc = best(pops, fits)
    sumtime = 0
    results = []
    for t in range(T):
        print("===================================")
        print("===================================")
        print("第{}次迭代后的结果".format(t+1))
        # 计时开始
        time_start = time.time()
        # step2 融合
        fuspops, fusFits, fusq = fusion(pops, q, w, input_x_data, input_y_data)

        # step3 分裂
        fispops, fisfits = fission(fuspops, fusFits, fusq, input_x_data, input_y_data)

        # step4 更新
        updatepops, updatefits = update(pops, fits, fispops, fisfits, N, L, best_individual, bestAcc, input_x_data,
                                        input_y_data)

        # 更新分裂概率
        if t > 2:
            difscore = 0.4 * (updatefits[:, 0] - fits[:, 0]) + 0.6 * (updatefits[:, 0] - bestAcc)
            diflen = 0.4 * ((updatefits[:, 1] - fits[:, 1]) / L) + 0.6 * ((updatefits[:, 1] - len(best_individual)) / L)
            # 计算分裂概率增量
            for i in range(len(difscore)):
                difq = 0
                if difscore[i] > 0:
                    difq = 0.4 / (1 + np.exp(-10 * diflen[i])) - 0.2
                else:
                    difq = 0.2 / (1 + np.exp(-10 * diflen[i])) + 0.2
                q[i] += difq
                if q[i] < 0.25:
                    q[i] = 0.25
                if q[i] > 0.8:
                    q[i] = 0.8

        # 准备下一轮迭代
        pops = updatepops
        fits = updatefits
        time_end = time.time()
        sumtime += (time_end - time_start)
        best_individual, bestAcc = best(pops, fits)
        best_ind = sorted(list(_flatten(best_individual)))
        best_ind = [i for i in best_ind if i != -1]
        print("当前种群个体数量:", len(pops))
        print("best_individual length:", len(best_ind))
        print("第{}次迭代花费的时间：{}，最好适应度={},最好的个体:{},".format(t, (time_end - time_start) / 60, bestAcc, best_ind))
        results.append([bestAcc, len(best_ind), len(best_ind) / L, best_ind])
    print("算法运行总用时：", sumtime)
    return results


# 迭代结果可视化
def plot(results, T):
    X = []
    Y = []
    for i in range(T):
        X.append(i)
        Y.append(results[i][0])

    plt.plot(X, Y)
    plt.show()


def avg_row_len(lst):
    return sum(len(row) for row in lst) / len(lst)


if __name__ == "__main__":
    # 参数设置
    T = 150  # 迭代次数
    N = 50  # 种群大小
    qvalue = 0.3
    w = 2
    dataset_path = './dataset/'
    dataset_lists = os.listdir(dataset_path)
    for dataset in dataset_lists:
        name = dataset[:-5]
        print('dataset:', name)
        data = pd.read_excel(dataset_path + dataset).values
        x_data, y_data = data[:, :-1], data[:, -1]
        ms = MinMaxScaler()
        X_data = ms.fit_transform(np.array(x_data))
        L = X_data.shape[1]
        for i in range(1):
            # 调用方法
            results = faf(N, L, T, qvalue, w, x_data, y_data.astype(int))
            frame = pd.DataFrame(data=[each_list for each_list in results], columns=['bestAcc', 'length',
                                                                                     'length_percent',
                                                                                     'best_individual'])
            frame.to_csv('./result/' + name + '_' + str(i) + '_result.csv')
