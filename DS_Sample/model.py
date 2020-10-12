
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import pickle
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score as R2
import matplotlib
import matplotlib.style as style

from funcs import *
from model_funcs import *

currDir = os.getcwd()
parent = os.path.abspath(os.path.join(currDir, os.pardir))

# quick load to avoid doing hourly averaging!
file = 'hourlyData'
f = open(os.path.join(parent, file+'.pkl'), 'rb')
processed = pickle.load(f)
f.close()

# quick load to avoid doing hourly averaging!
file = 'times'
f = open(os.path.join(parent, file+'.pkl'), 'rb')
newTimes = pickle.load(f)
f.close()

# steps ahead to forecast
steps = [1, 2, 3, 4, 5, 6]
# diff -> T/F
# whether target variable should be differenced
diff=False
# number of tests to run
n_tests = 1

for step in steps:
    
    print('\n')
    print(step, 'hour ahead prediction')
    tPrint = str(step)+'_step_target'
    pPrint = str(step)+'_step_prediction'
    ePrint = str(step)+'_step_error'
    testTimePrint = str(step)+'_step_test_time'
    
    # arrays for saved data
    targetArray = []
    predictionArray = []
    errorArray = []
    
    tot_rmse = np.zeros((n_tests, 1))

    for i in range(n_tests):
    
        trainingTimes, testingTimes, trainInputs, testInputs, trainTarget, testTarget, trainTargetVar, testTargetVar = dataForDirModel(processed, newTimes, step, diff)

        col = 0
        train_ref = trainInputs[:, col]
        test_ref = testInputs[:, col]
        true_train_target = trainTargetVar[:, col]
        true_test_target = testTargetVar[:, col]
        model_train_target = trainTarget[:, col]
        model_test_target = testTarget[:, col]

        feats = RandomForestRegressor(n_estimators=1000,
                                      min_samples_split=100,
                                      max_features=0.5,
                                      oob_score=True)
        feats.fit(trainInputs, model_train_target)
        pred = np.array(feats.predict(testInputs)).flatten()
        
        error = true_test_target-pred
        rmse = np.sqrt(error**2).mean()
        tot_rmse[i] = rmse
        
        targetArray.append(true_test_target)
        predictionArray.append(pred)
        errorArray.append(error)
        
        print('RMSE =', np.round(rmse, 3), 'm/s')
    
    print('Average RMSE =', np.round(np.mean(tot_rmse), 3), 'm/s')
    
