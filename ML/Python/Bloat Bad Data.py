import numpy as np
from sklearn.ensemble import GradientBoostingRegressor 
import matplotlib.pyplot as plt
import os.path
from os import path
from sklearn.neural_network import MLPRegressor

def LoadRawVariables():
    c2pt = []
    ts   = []
    taus = []
    xs   = []
    ys   = []
    zs   = []
    c3pt_S = []
    c3pt_V = []
    c3pt_A = []
    c2pt_OTHER = []
    sample_num = []
    
    
    for tau in range(0, 49, 8):
        for x in range(0, 25, 8):
            for y in range(0, 25, 8):
                for z in range(0, 25, 8):
                    for sample in range(748, 1421, 16):
                        fname = "../Data/T" + str(tau) + "/x" + str(x) + "y" + str(y) + "z" + str(z) + "/nuc3pt.dat." + str(sample)
                        if path.exists(fname):
                            with open(fname) as fp:
                                for i, line in enumerate(fp):
                                    if i >= 7 and i <= 70:           # The start of Gauss -> Point 2pt correlation functions
                                        c2pt_OTHER.append([float(x) for x in line.rstrip().split()[1:5]])
                                    if i >= 5182 and i <= 5245:      # The start of Gauss -> Gauss 2pt correlation functions
                                        c2pt.append([float(x) for x in line.rstrip().split()[1:5]])
                                        ts.append(i - 5182)
                                        taus.append(tau)
                                        sample_num.append(sample)
                                        xs.append(x)
                                        ys.append(y)
                                        zs.append(z)
                                    elif i >= 10154 and i <= 10217:
                                        c3pt_S.append([float(x) for x in line.rstrip().split()[1:5]])
                                    elif i >= 10229 and i <= 10292:
                                        c3pt_V.append([float(x) for x in line.rstrip().split()[1:5]])
                                    elif i >= 19979 and i <= 20042:
                                        c3pt_A.append([float(x) for x in line.rstrip().split()[1:5]])
                                    elif i > 20042:
                                        break
    
    return ts, taus, xs, ys, zs, c2pt, c3pt_S, c3pt_V, c3pt_A, c2pt_OTHER, sample_num

ts, taus, xs, ys, zs, c2pt, c3pt_S, c3pt_V, c3pt_A, c2pt_OTHER, sample_num = LoadRawVariables()

c2pt_factor_raw = sum(np.array(c2pt)) / len(c2pt)
N_factor = np.sqrt(c2pt_factor_raw[0] ** 2 + c2pt_factor_raw[1] ** 2)

for i in range(len(c2pt)):
    for j in range(len(c2pt[i])):
        c2pt_OTHER[i][j] /= N_factor
        c2pt[i][j] /= N_factor
    for j in range(len(c3pt_S[i])):
        c3pt_S[i][j] /= N_factor
        c3pt_V[i][j] /= N_factor
        c3pt_A[i][j] /= N_factor

## Features is a <# Data> x <Size Data> 2D array for all  lists of features (same for each time slice)
## Labels is a 64 x <# Data> 2D array for all 64 sets of lists of labels

features_unshifted = np.array([[taus[i]] + [c2pt[i + j][0] for j in range(64)] + [c2pt[i + j][1] for j in range(64)] for i in range(0, len(ts), 64)])
features = []

labels_S_up = np.array([[c3pt_S[i+j][0] for i in range(0, len(c3pt_S), 64)] for j in range(64)])
labels_A_up = np.array([[c3pt_A[i+j][0] for i in range(0, len(c3pt_A), 64)] for j in range(64)])
labels_V_up = np.array([[c3pt_V[i+j][0] for i in range(0, len(c3pt_V), 64)] for j in range(64)])

labels_S_down = np.array([[c3pt_S[i+j][2] for i in range(0, len(c3pt_S), 64)] for j in range(64)])
labels_A_down = np.array([[c3pt_A[i+j][2] for i in range(0, len(c3pt_A), 64)] for j in range(64)])
labels_V_down = np.array([[c3pt_V[i+j][2] for i in range(0, len(c3pt_V), 64)] for j in range(64)])

for i in range(len(features_unshifted)):
    shift = int(features_unshifted[i, 0])
    features.append(np.roll(features_unshifted[i, 1:], -shift))
    labels_S_up[:, i] = np.roll(labels_S_up[:, i], -shift)
    labels_A_up[:, i] = np.roll(labels_A_up[:, i], -shift)
    labels_V_up[:, i] = np.roll(labels_V_up[:, i], -shift)
    labels_S_down[:, i] = np.roll(labels_S_down[:, i], -shift)
    labels_A_down[:, i] = np.roll(labels_A_down[:, i], -shift)
    labels_V_down[:, i] = np.roll(labels_V_down[:, i], -shift)

features = np.array(features)

## Shuffling data - Holding off to check the writing indices
# perm = np.random.permutation(features.shape[0])
# np.take(features, perm, axis=0, out=features)
# np.take(labels_S_up, perm, axis=1, out=labels_S_up)
# np.take(labels_A_up, perm, axis=1, out=labels_A_up)
# np.take(labels_V_up, perm, axis=1, out=labels_V_up)
# np.take(labels_S_down, perm, axis=1, out=labels_S_down)
# np.take(labels_A_down, perm, axis=1, out=labels_A_down)
# np.take(labels_V_down, perm, axis=1, out=labels_V_down)

c2pt_footer = "ENDPROP\n"
c3pt_footer = "END_NUC3PT\n"
c2pt_header = """STARTPROP
MASSES:  1.000000e-03 1.000000e-03 1.000000e-03
SOURCE: GAUSS 70 600 0
SINK: GAUSS
MOM: 0 0 0
OPER: NUC_G5C_PP
"""
c3pt_V_header = """START_NUC3PT
MASSES:  1.000000e-03 1.000000e-03 1.000000e-03
SOURCE: GAUSS 70 600 0
SINK: GAUSS 9
SNK_MOM: 0 0 0
OPER: G4
OP_MOM: 0 0 0
FACT: 1.000000e+00 0.000000e+00
PROJ: PPAR
QUARKS:    up      down
"""
c3pt_S_header = """START_NUC3PT
MASSES:  1.000000e-03 1.000000e-03 1.000000e-03
SOURCE: GAUSS 70 600 0
SINK: GAUSS 9
SNK_MOM: 0 0 0
OPER: G0
OP_MOM: 0 0 0
FACT: 1.000000e+00 0.000000e+00
PROJ: PPAR
QUARKS:    up      down
"""
c3pt_A_header = """START_NUC3PT
MASSES:  1.000000e-03 1.000000e-03 1.000000e-03
SOURCE: GAUSS 70 600 0
SINK: GAUSS 9
SNK_MOM: 0 0 0
OPER: G5G3
OP_MOM: 0 0 0
FACT: 0.000000e+00 1.000000e+00
PROJ: PPAR_5Z
QUARKS:    up      down
"""

labelFrac = 0

labelEnd = int(len(labels_S_up[0]) * labelFrac)

X_train, Y_train_up, Y_train_down = features[:labelEnd], labels_S_up[:, :labelEnd], labels_S_down[:, :labelEnd]
X_test, Y_test_up, Y_test_down = features[labelEnd:], labels_S_up[:, labelEnd:], labels_S_down[:, labelEnd:]


### Vector Charge

X_train, Y_train_up, Y_train_down = features[:labelEnd], labels_V_up[:, :labelEnd], labels_V_down[:, :labelEnd]
X_test, Y_test_up, Y_test_down = features[labelEnd:], labels_V_up[:, labelEnd:], labels_V_down[:, labelEnd:]

for i in range(len(X_test)):
    fakeName = "../Data/FakeData/FakeData_x" + str(xs[64 * labelEnd + 64 * i]) + "y" + str(ys[64 * labelEnd + 64 * i]) + "z" + str(zs[64 * labelEnd + 64 * i]) + "samp" + str(sample_num[64 * labelEnd + 64 * i]) + ".txt"
    fakeDataFile = open(fakeName, "a")
    testImg = X_test[i]
    fakeDataFile.write(c3pt_V_header)
    for t in range(64):
        fakeDataFile.write(str(t) + " " + str(Y_test_up[t][i] * N_factor) + " 0.0 " + str(Y_test_down[t][i] * N_factor) + " 0.0\n")
    fakeDataFile.write(c3pt_footer)

### Axial Charge

X_train, Y_train_up, Y_train_down = features[:labelEnd], labels_A_up[:, :labelEnd], labels_A_down[:, :labelEnd]
X_test, Y_test_up, Y_test_down = features[labelEnd:], labels_A_up[:, labelEnd:], labels_A_down[:, labelEnd:]

for i in range(len(X_test)):
    fakeName = "../Data/FakeData/FakeData_x" + str(xs[64 * labelEnd + 64 * i]) + "y" + str(ys[64 * labelEnd + 64 * i]) + "z" + str(zs[64 * labelEnd + 64 * i]) + "samp" + str(sample_num[64 * labelEnd + 64 * i]) + ".txt"
    fakeDataFile = open(fakeName, "a")
    testImg = X_test[i]
    fakeDataFile.write(c3pt_A_header)
    for t in range(64):
        fakeDataFile.write(str(t) + " " + str(Y_test_up[t][i] * N_factor) + " 0.0 " + str(Y_test_down[t][i] * N_factor) + " 0.0\n")
    fakeDataFile.write(c3pt_footer)