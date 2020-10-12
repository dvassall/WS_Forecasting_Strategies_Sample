def fileLoad():
    import pickle
    import pandas
    import numpy
    import os
    
    currDir = os.getcwd()
    u = pickle.load(open('Spd_70m.pkl', 'rb'))
    d = pickle.load(open('Dir_70m.pkl', 'rb'))
    t = pickle.load(open('T_70m.pkl', 'rb'))
    
    return u, d, t

def varsOfInterest(u, d, t):
    import numpy as np
    import pandas as pd

    # double check that all data starts/ends
    # at the same time
    timeMatch(u, d, t)
    
    # fill in all missing times
    # all data made to have identical timestamps
    u2, d2, t2 = populateTimes(u, d, t)

    # get data of interest
    time = u2.index
    U = u2['Value']
    sigmaU = u2['Deviation']
    theta = d2['Value']
    temp = t2['Value']

    # concatenate data into one array
    newData = pd.concat([U, sigmaU, theta, temp], axis=1)
    newData.columns = ['U', 'sigmaU', 'Dir', 'Temp']
    
    return newData

def timeMatch(u, d, t):
    import numpy as np
    import pandas as pd
    bU, bD, bT = u['Time'].iloc[0], d['Time'].iloc[0], t['Time'].iloc[0]
    eU, eD, eT = u['Time'].iloc[-1], d['Time'].iloc[-1], t['Time'].iloc[-1]

    if (bU==bD) and (bU==bT) and (eU==eD) and (eU==eT):
        print('Start/end times match')
    else:
        raise Exception("Start/end times don't match!")
        
def populateTimes(u, d, t):
    import pandas as pd
    
    # reset index to timestamp - for upsampling func
    u2 = u.set_index('Time')
    d2 = d.set_index('Time')
    t2 = t.set_index('Time')
    
    # fill in all missing times - upsampled
    # asfreq() sets all missing data to NaNs
    uFull = u2.resample('10min', base=0, axis=0).asfreq()
    dFull = d2.resample('10min', base=0, axis=0).asfreq()
    tFull = t2.resample('10min', base=0, axis=0).asfreq()
    
    # replace NaN values with -999
    uFull = uFull.fillna(-999)
    dFull = dFull.fillna(-999)
    tFull = tFull.fillna(-999)
    
    return uFull, dFull, tFull


def hourlyAvgClean(npData, times, begin):
    import numpy as np
    import pandas as pd

    avgData = list()
    badTimes = list()
    newTimes = list()
    weirdTimes = list()
    for x in begin:

        # data window for single time period
        window = npData[x-5:x+1,:]
        # get count of bad values for each individual data source
        unique, counts = np.unique(np.argwhere(window==-999)[:, 1], return_counts=True)
        # if more than 50% of any given column's data is bad...
        if any(c>3 for c in counts):
            # list of bad times
            badTimes.append(x)
        else:
            # just in case something weird happens
            try:
                goodData = np.copy(window)
                goodData[goodData==-999] = np.nan
                avg = np.nanmean(goodData, axis=0)
                avgData.append(avg)
                newTimes.append(times[x])
            except:
                # anything goes wrong - NaNs
                avgData.append([np.nan]*4)
                weirdTimes.append(times[x])
    return avgData, newTimes, badTimes, weirdTimes

def hourlyCleaning(data):
    import pandas as pd
    import numpy as np
    
    # array of timestamps
    times = data.index
    
    # get minutes to start averaging
    minutes = np.array(times.minute)
    # get position index of all starting hours
    begin = np.argwhere(minutes==0).flatten()
    npData = np.array(data)
        
    # get hourly averages if >50% data is available for all relevant columns
    # else fill with nans
    avgData, newTimes, badTimes, weirdTimes = hourlyAvgClean(npData, times, begin)
    pdAvg = pd.DataFrame(avgData, columns=['U', 'sigma_U', 'Dir', 'Temp'])
    
    return begin, npData, pdAvg, newTimes, badTimes, weirdTimes

def processFinalData(avgData, newTimes):
    import numpy as np
    import pandas as pd

    times = pd.to_datetime(newTimes)

    u = np.array(avgData['U'])
    sigma = np.array(avgData['sigma_U'])
    temp = np.array(avgData['Temp'])
    ti = sigma/u
    direction = np.array(avgData['Dir'])
    hour = np.array(times.hour)
    day = np.array(times.dayofyear)

    # time of day arrays
    h1 = np.cos(hour*np.pi*2/24)
    h2 = np.sin(hour*np.pi*2/24)
    # seasonality arrays
    s1 = np.cos(day*np.pi*2/366)
    s2 = np.sin(day*np.pi*2/366)
    # direction arrays
    d1 = np.cos(direction*np.pi*2/360)
    d2 = np.sin(direction*np.pi*2/360)
    
    newData = np.stack((u, ti, d1, d2, temp, h1, h2, s1, s2), axis=1)
    pdNew = pd.DataFrame(newData, columns=['U', 'TI', 'DirNS', 'DirEW', 'T', 't1', 't2', 's1', 's2'],
                        index = times)
    
    return pdNew


def goodPeriods(newTimes, steps):
    import numpy as np
    import pandas as pd

    # steps = timesteps to look ahead for prediction

    # get list of times with requisite info
    pastTimes = pd.to_datetime(newTimes)
    currTimes = pastTimes+pd.DateOffset(hours=1)
    matchingCurr = np.argwhere(np.isin(currTimes, pastTimes)).flatten()
    pastTimes = pastTimes[matchingCurr]
    futureTimes = pastTimes+pd.DateOffset(hours=steps+1)
    matchingPred = np.argwhere(np.isin(futureTimes, pastTimes)).flatten()
    
    # final list of times (if single timestep) with data available at timesteps t & t+steps
    pastTimes = pastTimes[matchingPred]
    
    # extra step necessary for a multistep forecast - still need single step for training
    if steps != 1:
        singleStepTimes = pastTimes+pd.DateOffset(hours=2)
        matchingSingleStep = np.argwhere(np.isin(singleStepTimes, pastTimes)).flatten()
        # final list of times (multistep) with data available at timesteps t, t+1, & t+steps
        pastTimes = pastTimes[matchingSingleStep]
    
    # split into train & test sets
    trainLen = np.floor(len(pastTimes)*.9).astype(int)
    trainTimes = pastTimes[:trainLen]
    testTimes = pastTimes[trainLen:]
    
    return trainTimes, testTimes

def persistence(testInputs, multi_test_target, col):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score as R2
    
    pred = testInputs[:, col]
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
    
    return error
