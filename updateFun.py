import numpy as np
from function import func
from InitialPops import initpops
from tkinter import _flatten


def update(pops, fits, fispops, fisfits, N, L, best_individual, bestAcc, input_x_data, input_y_data):
    newpops = pops + fispops  # 拼接两个list
    newFits = np.concatenate((fits, fisfits))  # 拼接两个array
    updatepops = []
    updatepops.append(best_individual)
    updatefits = np.array([[bestAcc, len(best_individual)]])
    """第一步：去除重复合团"""
    for i in range(len(newpops)):
        if newpops[i] not in updatepops:
            updatepops.append(newpops[i])
            updatefits = np.append(updatefits, newFits[i].reshape(1, 2), 0)
        else:
            continue

    """第二步，判断现在群体规模与初始群体规模N的大小关系，增加或者删除一些合团"""
    # 1 先从当前种群中抽出2/3N个合团作为 updatepop
    score_pops = zip(updatefits, updatepops)
    sorted_score_pops = sorted(score_pops, reverse=True, key=lambda x: x[0][0])
    tuple_fits, tuple_pops, = zip(*sorted_score_pops)
    updatefits = np.array(tuple_fits)[:int((2 * N) / 3)]
    updatepops = list(tuple_pops)[:int((2 * N) / 3)]

    # 2 补充新合团，使得规模保持为N
    new_pops = initpops(N-len(updatepops), L)
    updatepops = updatepops + new_pops

    for ind in new_pops:
        ind = list(_flatten(ind))
        ind = [i for i in ind if i != -1]
        fit = np.array(func(ind, input_x_data, input_y_data))
        updatefits = np.append(updatefits, fit.reshape(1, 2), 0)

    return updatepops, updatefits
