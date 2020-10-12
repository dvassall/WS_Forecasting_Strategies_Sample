
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
diff=True
# number of tests to run
n_tests = 1

for step in steps:

    print('\n')
    print(step, 'hours ahead')
    tPrint = str(step)+'_step_target'
    pPrint = str(step)+'_step_prediction'
    ePrint = str(step)+'_step_error'
    testTimePrint = str(step)+'_step_test_time'
    
    # arrays for saved data
    targetArray = []
    predictionArray = []
    errorArray = []
    
    avg_rmse = np.zeros((n_tests, 1))
    
    for i in range(n_tests):

        trainTimes, testTimes, trainInputs, testInputs, trainTarget, testTarget, train_target_var, test_target_var, multi_train_target, multi_test_target = dataForNwpModel(processed, newTimes, step, diff)

        # produce results for test set
        testing=False
        # save model parameters for future use
        saveModel=False

        Umodel, TImodel, DirNSmodel, DirEWmodel, Tmodel = models(trainInputs, testInputs, trainTarget, testTarget, train_target_var, test_target_var, step, diff, testing, saveModel)

        col=0
        truePred = prediction(Umodel, TImodel, DirNSmodel, DirEWmodel, Tmodel, testInputs, testTimes, step, diff)
        
        pred = truePred[:,col]
        target = multi_test_target[:, col]
        error = target-pred
        rmse = np.sqrt(error**2).mean()
        avg_rmse[i] = rmse
        
        targetArray.append(target)
        predictionArray.append(pred)
        errorArray.append(error)

        print('RMSE =', np.round(rmse, 3), 'm/s')
        
    print('Average RMSE =', np.round(np.mean(avg_rmse), 3), 'm/s')
