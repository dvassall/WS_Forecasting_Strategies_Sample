def dirModelData(processed, trainTimes, testTimes, steps, diff):
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

    # get necessary sorted data
    pastTrainData = np.array(processed.iloc[pastTrainingTimes, :])
    pastTestData = np.array(processed.iloc[pastTestingTimes, :])
    currentTrainData = np.array(processed.iloc[currentTrainingTimes, :])
    currentTestData = np.array(processed.iloc[currentTestingTimes, :])
    # for pca...
    pastTrainUData = np.array(processed['U'].iloc[pastTrainingTimes])
    currentTrainUData = np.array(processed['U'].iloc[currentTrainingTimes])
    pastTestUData = np.array(processed['U'].iloc[pastTestingTimes])
    currentTestUData = np.array(processed['U'].iloc[currentTestingTimes])


    ###### input arrays #######
    trainInputs = np.concatenate((currentTrainData, pastTrainData), axis=1)
    testInputs = np.concatenate((currentTestData, pastTestData), axis=1)

    ###### target arrays #######
    # (true - use for comparison after running model)
    trainTargetVar = np.array(processed.iloc[futureTrainingTimes, :])
    testTargetVar = np.array(processed.iloc[futureTestingTimes, :])
    
    ###### ML target arrays ########
    if diff:
        # (Model predicting difference between current and future period)
        trainTarget = trainTargetVar-currentTrainData
        testTarget = testTargetVar-currentTestData
    else:
        # Model predicting absolute value of target variable
        trainTarget = trainTargetVar
        testTarget = testTargetVar

    return trainInputs, testInputs, trainTarget, testTarget, trainTargetVar, testTargetVar


def dataForDirModel(processed, newTimes, steps, diff):
    import numpy as np
    import pandas as pd
    from funcs import goodPeriods
    
    # get list of times (t-1; prev. time) that have data available
    # at time (t; current time) & (t+n; prediction time)
    trainTimes, testTimes = goodPeriods(newTimes, steps)
    
    trainInputs, testInputs, trainTarget, testTarget, trainTargetVar, testTargetVar = dirModelData(processed, trainTimes, testTimes, steps, diff)
    
    return trainTimes, testTimes, trainInputs, testInputs, trainTarget, testTarget, trainTargetVar, testTargetVar
