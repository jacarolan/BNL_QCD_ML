#!/usr/bin/env python

import numpy as np
from sklearn import ensemble

# Read data
data = np.load('data-axial.npy', allow_pickle=True).tolist()
# data['train'/'test']['input'/'output']
# inputs are vectors with 40 elements, and outputs are single numbers

print(data['train']['input'].shape)
print(data['train']['output'].shape)
print(data['test']['input'].shape)
print(data['test']['output'].shape)

# Train regression algorithm
regr = ensemble.GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=None)
regr.fit(data['train']['input'], data['train']['output'])

# Make predictions
y_pred = regr.predict(data['test']['input'])

# Calculate error
pred_err = data['test']['output'] - y_pred
pred_quality = np.std(pred_err) / np.std(data['train']['output'])

# Here pred_quality (prediction quality) is square-root of 
# the mean square error normalized by the standard deviation
# of raw data. It becomes 0 for a perfect prediction, and 
# pred_quality > 1 indicates no prediction.

print("Prediction quality = ", pred_quality)
# Expected output: "Prediction quality =  0.5220197021694335"
