#!/usr/bin/env python

import numpy as np
from sklearn import ensemble
import matplotlib.pyplot as plt

# Read data
data = np.load('data-axial.npy', allow_pickle=True).tolist()
# data['train'/'test']['input'/'output']
# inputs are vectors with 40 elements, and outputs are single numbers

print(data['train']['input'].shape)
print(data['train']['output'].shape)
print(data['test']['input'].shape)
print(data['test']['output'].shape)

# Train regression algorithm
regr = ensemble.GradientBoostingRegressor(learning_rate=0.05, n_estimators=50, max_depth=3)
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

X_test = data['test']['input']
Y_test = data['test']['output']

trials = 0
errors = []
raw_RMS = 0
ML_samples = []
DM_samples = []
for i in range(len(X_test)):
    testImg = X_test[i]
    testLabel = Y_test[i]
    pred = regr.predict([testImg])
    errors.append(pred - testLabel)
    ML_samples.append(pred[0])
    DM_samples.append(testLabel)
    
print("Prediction quality:", np.std(errors) / np.std(Y_test))

plt.hist(DM_samples, bins=20)
plt.hist(ML_samples, bins=20)
plt.legend(["Raw Data", "ML Prediction"])
plt.xlabel("Real part of c3pt")
plt.ylabel("Prediction count")
plt.show()