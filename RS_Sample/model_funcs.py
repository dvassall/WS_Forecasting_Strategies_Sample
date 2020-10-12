def nwpModelData(processed, trainTimes, testTimes, steps, diff):
    import numpy as np
    import pandas as pd
    # get training and testing data - inputs & targets

    # list of times from data index
    matchTimes = processed.index

    # get list of matching times for each data source
    pastTrainingTimes = matchTimes.isin(trainTimes)
    pastTestingTimes = matchTimes.isin(testTimes)
    currentTrainingTimes = matchTimes.isin(trainTimes+pd.DateOffset(hours=1))
    currentTestingTimes = matchTimes.isin(testTimes+pd.DateOffset(hours=1))
    futureTrainingTimes = matchTimes.isin(trainTimes+pd.DateOffset(hours=1+steps))
    futureTestingTimes = matchTimes.isin(testTimes+pd.DateOffset(hours=1+steps))
    
    singleTrainingTimes = matchTimes.isin(trainTimes+pd.DateOffset(hours=2))
    singleTestingTimes = matchTimes.isin(testTimes+pd.DateOffset(hours=2))

    # get necessary sorted data
    pastTrainData = np.array(processed.iloc[pastTrainingTimes, :])
    pastTestData = np.array(processed.iloc[pastTestingTimes, :])
    currentTrainData = np.array(processed.iloc[currentTrainingTimes, :])
    currentTestData = np.array(processed.iloc[currentTestingTimes, :])

    ###### input arrays #######
    trainInputs = np.concatenate((currentTrainData, pastTrainData), axis=1)
    testInputs = np.concatenate((currentTestData, pastTestData), axis=1)

    ###### n_step target arrays #######
    # (true - use for comparison after running model)
    finalTrainTargetVar = np.array(processed.iloc[futureTrainingTimes, :])
    finalTestTargetVar = np.array(processed.iloc[futureTestingTimes, :])
    
    ###### single step target arrays #######
    singleTrainTargetVar = np.array(processed.iloc[singleTrainingTimes, :])
    singleTestTargetVar = np.array(processed.iloc[singleTestingTimes, :])
    
    ###### single step ML target arrays #######
    if diff:
        # (Model predicting difference between current and future period)
        trainTarget = singleTrainTargetVar-currentTrainData
        testTarget = singleTestTargetVar-currentTestData
        
    else:
        # model simply predicting absolute values
        trainTarget = singleTrainTargetVar
        testTarget = singleTestTargetVar

    return trainInputs, testInputs, trainTarget, testTarget, singleTrainTargetVar, singleTestTargetVar, finalTrainTargetVar, finalTestTargetVar


def dataForNwpModel(processed, newTimes, steps, diff):
    import numpy as np
    import pandas as pd
    from funcs import goodPeriods
    
    # get list of times (t-1; prev. time) that have data available
    # at time (t; current time) & (t+n; prediction time)
    trainTimes, testTimes = goodPeriods(newTimes, steps)
    trainInputs, testInputs, trainTarget, testTarget, singleTrainTargetVar, singleTestTargetVar, finalTrainTargetVar, finalTestTargetVar = nwpModelData(processed, trainTimes, testTimes, steps, diff)
    
    return trainTimes, testTimes, trainInputs, testInputs, trainTarget, testTarget, singleTrainTargetVar, singleTestTargetVar, finalTrainTargetVar, finalTestTargetVar

def trainModel(trainInputs, testInputs, trainTarget, testTarget,
               train_target_var, test_target_var, col, steps,
               diff, testing=False, saveModel=False):
    import numpy as np
    import pandas as pd
    from joblib import dump, load
    from sklearn.ensemble import RandomForestRegressor
    import os

    # create model
    # col: 0=U, 1=TI, 2=DirNS, 3=DirEW, 4=T
    true_train_target = train_target_var[:, col]
    true_test_target = test_target_var[:, col]
    model_train_target = trainTarget[:, col]
    model_test_target = testTarget[:, col]
    train_ref = trainInputs[:, col]
    test_ref = testInputs[:, col]
    
    feats = RandomForestRegressor(n_estimators=1000,
                                  min_samples_split=100,
                                  max_features=0.5,
                                  oob_score=True)
    feats.fit(trainInputs, model_train_target)
    
    # produces test set stats & predictions
    if testing:
        prediction = testStats(feats, testInputs, test_ref, true_test_target, diff)
    else:
        prediction = []
        
    # saves the model parameters as a .joblib file
    if saveModel:
        if col==0:
            newFile = os.path.join('Umodel_'+str(steps)+'_steps.joblib')
        elif col==1:
            newFile = os.path.join('TImodel_'+str(steps)+'_steps.joblib')
        elif col==2:
            newFile = os.path.join('DirNSmodel_'+str(steps)+'_steps.joblib')
        elif col==3:
            newFile = os.path.join('DirEWmodel_'+str(steps)+'_steps.joblib')
        elif col==4:
            newFile = os.path.join('Tmodel_'+str(steps)+'_steps.joblib')
        else:
            raise Exception('You entered the wrong column')
        dump(feats, newFile)
        
    return feats, prediction
    

def testStats(feats, testInputs, test_ref, true_test_target, diff):
    import numpy as np
    from sklearn.metrics import r2_score as R2
    
    pred = np.array(feats.predict(testInputs)).flatten()
    if diff:
        prediction = pred+test_ref
    else:
        prediction = pred

    error = true_test_target-prediction
    mse = (error**2).mean()
    rmse = np.sqrt(error**2).mean()
    stde = error.std()
    uncertainty = stde*100/(true_test_target.mean())
    r = np.round(R2(true_test_target, prediction)*100, 1)

    print('RMSE (MAE) =', np.round(rmse, 3), 'm/s')
    print('MSE =', np.round(mse, 3), 'm/s')
    print('Uncertainty =', np.round(uncertainty, 2), '%')
    print('R^2 =', np.round(r, 1), '%')
    
    return prediction


def models(trainInputs, testInputs, trainTarget, testTarget, train_target_var, test_target_var, steps, diff, testing=False, saveModel=False):

    # each variable has its own column
    Ucol, TIcol, DirNScol, DirEWcol, Tcol = 0, 1, 2, 3, 4

    Umodel, Uprediction = trainModel(trainInputs, testInputs, trainTarget, testTarget, train_target_var, test_target_var, Ucol, steps, diff, testing, saveModel)
    print('U model complete')
    TImodel, TIprediction = trainModel(trainInputs, testInputs, trainTarget, testTarget, train_target_var, test_target_var, TIcol, steps, diff, testing, saveModel)
    print('TI model complete')
    DirNSmodel, DirNSprediction = trainModel(trainInputs, testInputs, trainTarget, testTarget, train_target_var, test_target_var, DirNScol, steps, diff, testing, saveModel)
    print('DirNS model complete')
    DirEWmodel, DirEWprediction = trainModel(trainInputs, testInputs, trainTarget, testTarget, train_target_var, test_target_var, DirEWcol, steps, diff, testing, saveModel)
    print('DirEW model complete')
    Tmodel, Tprediction = trainModel(trainInputs, testInputs, trainTarget, testTarget, train_target_var, test_target_var, Tcol, steps, diff, testing, saveModel)
    print('T model complete')

    return Umodel, TImodel, DirNSmodel, DirEWmodel, Tmodel

def createNewTimes(testTimes, step):
    import pandas as pd
    import numpy as np
    
    t = testTimes+pd.DateOffset(hours=step)
    newHours = np.array(t.hour)
    newDays = np.array(t.dayofyear)
    h1 = np.cos(newHours*np.pi*2/24)
    h2 = np.sin(newHours*np.pi*2/24)
    s1 = np.cos(newDays*np.pi*2/366)
    s2 = np.sin(newDays*np.pi*2/366)
    
    season_stepped = np.stack((h1, h2, s1, s2), axis=1)
    
    return season_stepped


def prediction(Umodel, TImodel, DirNSmodel, DirEWmodel, Tmodel, testInputs, testTimes, steps, diff):
    import pandas as pd
    import numpy as np
    
    testIn = testInputs.copy()

    for step in range(steps):
        # predict all variables at next step
        predU, predTI, predDirNS, predDirEW, predT = Umodel.predict(testIn), TImodel.predict(testIn), DirNSmodel.predict(testIn), DirEWmodel.predict(testIn), Tmodel.predict(testIn)

        # concatenate all variables to use as inputs
        if diff:
            truePred = np.stack((predU, predTI, predDirNS, predDirEW, predT), axis=1)+testIn[:, :5]
        else:
            truePred = np.stack((predU, predTI, predDirNS, predDirEW, predT), axis=1)

        # get new time data for next step
        season_stepped = createNewTimes(testTimes, step)

        # concatenate all info for next step
        newVec = np.concatenate((truePred, season_stepped), axis=1)

        # replace old info with new info
        testIn = testIn[:, :9]
        testIn = np.concatenate((newVec, testIn), axis=1)
        
    return truePred


def stats(prediction, multi_test_target, col):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score as R2
    
    pred = prediction[:,col]
    target = multi_test_target[:, col]
    error = target-pred
    mse = (error**2).mean()
    rmse = np.sqrt(error**2).mean()
    stde = error.std()
    uncertainty = stde*100/(target.mean())
    r = np.round(R2(target, pred)*100, 1)

    print('RMSE (MAE) =', np.round(rmse, 3), 'm/s')
    print('MSE =', np.round(mse, 3), 'm/s')
    print('Uncertainty =', np.round(uncertainty, 2), '%')
    print('R^2 =', np.round(r, 1), '%')


