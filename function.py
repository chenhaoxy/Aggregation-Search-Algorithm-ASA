"""
优化的目标函数
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def func(individual, x_data, y_data):
    if len(individual) != 0:
        knn = KNeighborsClassifier(n_neighbors=5)
        # cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
        scores = cross_val_score(knn, x_data[:, individual], y_data, cv=5, scoring='accuracy')
        score = scores.mean()
        return score, len(individual)

    else:
        return 0.0

