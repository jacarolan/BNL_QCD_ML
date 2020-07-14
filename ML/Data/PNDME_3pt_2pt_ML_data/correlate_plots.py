#!/usr/bin/env python

import numpy as np
from sklearn import ensemble
from scipy.stats import pearsonr

# Read data
data = np.load('data-axial.npy', allow_pickle=True).tolist()
# data['train'/'test']['input'/'output']
# inputs are vectors with 40 elements, and outputs are single numbers

c2pt_avgs = np.array([sum(x) / len(x) for x in data['train']['input']])

corr, _ = pearsonr(c2pt_avgs, data['train']['output'])

print("Axial correlation at 10a separation: ", np.sqrt(corr))

# Read data
data = np.load('data-vector.npy', allow_pickle=True).tolist()
# data['train'/'test']['input'/'output']
# inputs are vectors with 40 elements, and outputs are single numbers

c2pt_avgs = np.array([sum(x) / len(x) for x in data['train']['input']])

corr, _ = pearsonr(c2pt_avgs, data['train']['output'])

print("Vector correlation at 10a separation: ", np.sqrt(corr))