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
    
    
    for tau in range(0, 49, 8):
        for x in range(0, 25, 8):
            for y in range(0, 25, 8):
                for z in range(0, 25, 8):
                    for sample in range(748, 800, 16): #1421, 16):
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
    
    return ts, taus, xs, ys, zs, c2pt, c3pt_S, c3pt_V, c3pt_A, c2pt_OTHER

ts, taus, xs, ys, zs, c2pt, c3pt_S, c3pt_V, c3pt_A, c2pt_OTHER = LoadRawVariables()

c2pt_factor_raw = sum(np.array(c2pt)) / len(c2pt)
N_factor = np.sqrt(c2pt_factor_raw[0] ** 2 + c2pt_factor_raw[1] ** 2)

for i in range(len(c2pt)):
    for j in range(len(c2pt[i])):
        c2pt[i][j] /= N_factor
    for j in range(len(c3pt_S[i])):
        c3pt_S[i][j] /= N_factor
        c3pt_V[i][j] /= N_factor
        c3pt_A[i][j] /= N_factor
        c2pt_OTHER[i][j] /= N_factor

# features = np.array([np.array([ts[i], taus[i], xs[i], ys[i], zs[i], c2pt[i][0], c2pt[i][1]]) for i in range(len(ts))])
features_unshifted = np.array([[taus[i]] + [c2pt[i + j][0] for j in range(64)] + [c2pt[i + j][1] for j in range(64)] for i in range(0, len(ts), 64)])
features = []
for f in features_unshifted:
    shift = int(f[0])
    features.append(np.roll(f[1:], -shift))

features = np.array(features)

labels_S_up = np.array([sum(c3pt_S[i:i+64][0]) / 64 for i in range(0, len(c3pt_S), 64)])
labels_A_up = np.array([sum(c3pt_A[i:i+64][0]) / 64 for i in range(0, len(c3pt_A), 64)])
labels_V_up = np.array([sum(c3pt_V[i:i+64][0]) / 64 for i in range(0, len(c3pt_V), 64)])

labels_S_down = np.array([sum(c3pt_S[i:i+64][2]) / 64 for i in range(0, len(c3pt_S), 64)])
labels_A_down = np.array([sum(c3pt_A[i:i+64][2]) / 64 for i in range(0, len(c3pt_A), 64)])
labels_V_down = np.array([sum(c3pt_V[i:i+64][2]) / 64 for i in range(0, len(c3pt_V), 64)])

labelFrac = 0.5
BCFrac = 0.1

c2pt_footer = "ENDPROP\n"
c3pt_footer = "END_NUC3PT\n"
c2pt_header = """STARTPROP
MASSES:  1.000000e-03 1.000000e-03 1.000000e-03
SOURCE: GAUSS 70 600 0 
SINK: POINT
MOM: 0 0 0
OPER: NUC_G5C_PP5
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



labelEnd = int(len(labels_S_up) * labelFrac)
BCEnd    = int(len(labels_S_up) * (BCFrac + labelFrac))

### Scalar Charge

X_train, Y_train_up, Y_train_down = features[:labelEnd], labels_S_up[:labelEnd], labels_S_down[:labelEnd]
X_bc, Y_bc_up, Y_bc_down = features[labelEnd:BCEnd], labels_S_up[labelEnd:BCEnd], labels_S_down[labelEnd:BCEnd]
X_test, Y_test_up, Y_test_down = features[BCEnd:], labels_S_up[BCEnd:], labels_S_down[BCEnd:]

gbr_up = GradientBoostingRegressor(learning_rate=0.05, n_estimators=50, max_depth=3)
gbr_up.fit(X_train, Y_train_up)

y_bc_pred = gbr_up.predict(X_bc)

biasCrxn_up = np.average(Y_bc_up - y_bc_pred)

gbr_down = GradientBoostingRegressor(learning_rate=0.05, n_estimators=50, max_depth=3)
gbr_down.fit(X_train, Y_train_down)

y_bc_pred = gbr_down.predict(X_bc)

biasCrxn_down = np.average(Y_bc_down - y_bc_pred)

trials = 0
errors = []
raw_RMS = 0
ML_samples = []
DM_samples = []
for i in range(len(X_test)):
    testImg = X_test[i]
    pred_up = gbr_up.predict([testImg])[0] + biasCrxn_up
    pred_down = gbr_down.predict([testImg])[0] + biasCrxn_down
    fakeName = "../Data/FakeData/FakeData" + str(i) + ".txt"
    if not os.path.exists(fakeName):
    	with open(fakeName, 'w+'): pass
    fakeDataFile = open(fakeName, "r+")
    fakeDataFile.truncate(0)
    fakeDataFile.write(c2pt_header)
    for j in range(64):
    	fakeDataFile.write(str(j) + " " + str(X_test[i][j]) + " " + str(X_test[i][j + 64]) + "\n")
    fakeDataFile.write(c2pt_footer)
    fakeDataFile.write(c3pt_S_header)
    for j in range(64):
    	fakeDataFile.write(str(j) + " " + str(pred_up) + " 0.0 " + str(pred_down) + " 0.0\n")
    fakeDataFile.write(c3pt_footer)

### Vector Charge

X_train, Y_train_up, Y_train_down = features[:labelEnd], labels_V_up[:labelEnd], labels_V_down[:labelEnd]
X_bc, Y_bc_up, Y_bc_down = features[labelEnd:BCEnd], labels_V_up[labelEnd:BCEnd], labels_V_down[labelEnd:BCEnd]
X_test, Y_test_up, Y_test_down = features[BCEnd:], labels_V_up[BCEnd:], labels_V_down[BCEnd:]

gbr_up = GradientBoostingRegressor(learning_rate=0.05, n_estimators=50, max_depth=3)
gbr_up.fit(X_train, Y_train_up)

y_bc_pred = gbr_up.predict(X_bc)

biasCrxn_up = np.average(Y_bc_up - y_bc_pred)

gbr_down = GradientBoostingRegressor(learning_rate=0.05, n_estimators=50, max_depth=3)
gbr_down.fit(X_train, Y_train_down)

y_bc_pred = gbr_down.predict(X_bc)

biasCrxn_down = np.average(Y_bc_down - y_bc_pred)

trials = 0
errors = []
raw_RMS = 0
ML_samples = []
DM_samples = []
for i in range(len(X_test)):
    testImg = X_test[i]
    pred_up = gbr_up.predict([testImg])[0] + biasCrxn_up
    pred_down = gbr_down.predict([testImg])[0] + biasCrxn_down
    fakeName = "../Data/FakeData/FakeData" + str(i) + ".txt"
    fakeDataFile = open(fakeName, "a")
    fakeDataFile.write(c3pt_V_header)
    for j in range(64):
    	fakeDataFile.write(str(j) + " " + str(pred_up) + " 0.0 " + str(pred_down) + " 0.0\n")
    fakeDataFile.write(c3pt_footer)

### Axial Charge

X_train, Y_train_up, Y_train_down = features[:labelEnd], labels_A_up[:labelEnd], labels_A_down[:labelEnd]
X_bc, Y_bc_up, Y_bc_down = features[labelEnd:BCEnd], labels_A_up[labelEnd:BCEnd], labels_A_down[labelEnd:BCEnd]
X_test, Y_test_up, Y_test_down = features[BCEnd:], labels_A_up[BCEnd:], labels_A_down[BCEnd:]

gbr_up = GradientBoostingRegressor(learning_rate=0.05, n_estimators=50, max_depth=3)
gbr_up.fit(X_train, Y_train_up)

y_bc_pred = gbr_up.predict(X_bc)

biasCrxn_up = np.average(Y_bc_up - y_bc_pred)

gbr_down = GradientBoostingRegressor(learning_rate=0.05, n_estimators=50, max_depth=3)
gbr_down.fit(X_train, Y_train_down)

y_bc_pred = gbr_down.predict(X_bc)

biasCrxn_down = np.average(Y_bc_down - y_bc_pred)

trials = 0
errors = []
raw_RMS = 0
ML_samples = []
DM_samples = []
for i in range(len(X_test)):
    testImg = X_test[i]
    pred_up = gbr_up.predict([testImg])[0] + biasCrxn_up
    pred_down = gbr_down.predict([testImg])[0] + biasCrxn_down
    fakeName = "../Data/FakeData/FakeData" + str(i) + ".txt"
    fakeDataFile = open(fakeName, "a")
    fakeDataFile.write(c3pt_V_header)
    for j in range(64):
    	fakeDataFile.write(str(j) + " " + str(pred_up) + " 0.0 " + str(pred_down) + " 0.0\n")
    fakeDataFile.write(c3pt_footer)
