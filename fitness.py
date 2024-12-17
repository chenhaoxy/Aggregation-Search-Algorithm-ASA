"""
种群或个体的适应度 
"""
import numpy as np
from function import func


def fitness(pops, x_data, y_data):
    fits = []
    for i, pop in enumerate(pops):
        feature_list = np.array(pop).flatten()
        feature_list = [i for i in feature_list if i != -1]
        fitness_value = func(feature_list, x_data, y_data)
        fits.append(fitness_value)
    return np.array(fits)
