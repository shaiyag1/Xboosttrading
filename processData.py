from __future__ import division
import time
import sys, getopt, traceback
import datetime
import ast
import os.path
import calendar
import xlsxwriter
import numpy as np
import copy
import math
from collections import defaultdict
from poloniex import poloniex
from numpy import loadtxt
from xgboost import XGBClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pandas as pa
import matplotlib.pyplot as plt


def testSellOpportunity (bid,historicalData,i):
    upMarkDiff = 1
    upMark = 0.0
    stopLoss1Trigger = False
    sellThresh = 0.99
    stopLoss1 = 1.01
    stopLoss2 = 0.95
    ROItresh  =1.05


    if bid == 0:
        return 0


    for row in historicalData[i:]:
        newPrice = float(row['close'])
        if upMark > 0:
            if newPrice > upMark:
                upMark = newPrice
                upMarkDiff = 1
            else:
                upMarkDiff = newPrice / upMark
        else:
            upMark = newPrice
            upMarkDiff = 1

        if bid > 0:
            diff = newPrice / bid
        else:
            diff = 0

        if diff > stopLoss1:
            stopLoss1Trigger = True

        if (upMarkDiff <= sellThresh or (diff < stopLoss1 and stopLoss1Trigger) or diff < stopLoss2):
            # Sell trigger reached
            # if newPrice/bid > 1.05:
            #     print "bid: %.12f selling price: %.12f diff diff: %.2f" % (bid, newPrice, newPrice / bid)
            return (newPrice/bid > ROItresh)
    return 0 # in case we get here

def testSellOpportunityEX (historicalData,i,historicalDataPA):
    upMarkDiff = 1
    upMark = 0.0
    stopLoss1Trigger = False
    sellThresh = 0.99
    stopLoss1 = 1.01
    stopLoss2 = 0.95
    ROItresh  =1.07
    futurerange=12*4

    bid = historicalDataPA[i]
    if bid == 0:
        return 0

    #closedata = pa.DataFrame(historicalData)
    #closedataarray = closedata['close'].as_matrix()
    till =min(i+futurerange,np.size(historicalDataPA))
    if(till<0.5*futurerange):
        return 0
    max_index_arr=np.argmax(historicalDataPA[i:till])
    max_index_arr=max_index_arr+i
    if(np.max(historicalDataPA[i:till])<ROItresh*bid):
        return 0
    if(np.max(historicalDataPA[i:till])!=historicalDataPA[max_index_arr]):
        for j in xrange(i,till):
            if(np.max(historicalDataPA[i:till])== historicalDataPA[j]):
                print(" Max index ", j ,historicalDataPA[j] )
        print('  %d   %10f     %10f    %10f ',i,bid, max_index_arr, np.max(historicalDataPA[i:till]),historicalDataPA[max_index_arr])
        input('Wrong Assumption')

    # print(max_index_arr)
    # print(type(max_index_arr))

    # for j in max_index_arr:
    #     if(np.min(closedataarray[i:j])/bid<sellThresh):
    #         return 0
    # Not accurate, we should find thar drastic min doesn't come befor the peak

    if(np.size(max_index_arr)==1):
        min_index_arr = np.argmin(historicalDataPA[i:till])+i
        if(np.size(min_index_arr) ==1):
            if(min_index_arr < max_index_arr and (np.min(historicalDataPA[min_index_arr]/bid)<sellThresh) ):
                return 0
        else:
            return 0
    else:
        return 0
    #ind buy point %s  %s  %s ",i,bid,historicalDataPA[max_index_arr])
    return 1




def getHist(coin,startTime,endTime):
    historicalData = {}

    ### read history directly from poloniex exchange
    conn = poloniex('', '')
    historicalData[coin] = conn.api_query("returnChartData",
                    {"currencyPair": coin, "start": startTime, "end": endTime, "period": 300})

    ### read history from previously saved file
    # with open('./hist/' + coin, 'r') as f:
    #     s = f.read()
    #     historicalData[coin] = ast.literal_eval(s)
    #     f.close()

    return historicalData


def createFeatures (f,ticker, movingPeriod,lastEma):

    b = np.array(ticker)
    rolling_max = max(b[:-1])
    f.write ("%.12f," % rolling_max)
    rolling_max = max(b[:-2])
    f.write ("%.12f," % rolling_max)
    rolling_max = max(b[-50:-1])
    f.write ("%.12f," % rolling_max)
    rolling_max = max(b[-50:-2])
    f.write ("%.12f," % rolling_max)
    rolling_max = max(b[-75:-1])
    f.write ("%.12f," % rolling_max)
    rolling_max = max(b[-75:-2])
    f.write ("%.12f," % rolling_max)

    for i in range(0,-60, -5):
        rolling_mean = float("{0:.12f}".format(np.mean(b[i:])))
        f.write("%.12f," % rolling_mean)
        rolling_std = float("{0:.12f}".format(np.std(b[i:], ddof=0)))
        f.write ("%.12f," % rolling_std)
        for j in 0.382,1,2,3,4:
            upper_band = float("{0:.12f}".format(rolling_mean + (rolling_std * j)))
            f.write("%.12f," % upper_band)
            lower_band = float("{0:.12f}".format(rolling_mean - (rolling_std * j)))
            f.write("%.12f," % lower_band)

    #EMA
    if lastEma == 0:
        ema = float("{0:.12f}".format(np.mean(b)))
        f.write("%.12f," % ema)
        f.write("%.12f," % ema)
        f.write("%.12f," % ema)
    else:
        ema = (float(ticker[-1]) * 2 / (5 + 1) + lastEma * (1 - 2 / (5 + 1)))
        f.write("%.12f," % ema)
        ema = (float(ticker[-1]) * 2 / (10 + 1) + lastEma * (1 - 2 / (10 + 1)))
        f.write("%.12f," % ema)
        ema = (float(ticker[-1]) * 2 / (20 + 1) + lastEma * (1 - 2 / (20 + 1)))
        f.write("%.12f," % ema)

    return ema

def createFeaturesEX(f, histdata, movingPeriod,lastEma):
    ticker=histdata[:,0]
    rolling_max = max(ticker[:-1])
    f.write ("%.12f," % rolling_max)
    rolling_max = max(ticker[:-2])
    f.write ("%.12f," % rolling_max)
    rolling_max = max(ticker[-50:-1])
    f.write ("%.12f," % rolling_max)
    rolling_max = max(ticker[-50:-2])
    f.write ("%.12f," % rolling_max)
    rolling_max = max(ticker[-75:-1])
    f.write ("%.12f," % rolling_max)
    rolling_max = max(ticker[-75:-2])
    f.write ("%.12f," % rolling_max)



def coinstat(coin,dt):
    coinstat.counter=coinstat.counter+1

    print("Coin Stat "+coin)
    print(dt[588:600])
    print("Max       " ,np.max(dt))
    print("Min       " ,np.min(dt))
    print("mean      " ,np.mean(dt))
    print("STD       "   ,np.std(dt))



    print("Start Val ",dt[0])
    print("End Val   ",dt[dt.shape[0]-1])
    print("Std Ratio: " ,np.std(dt)/np.mean(dt))
    dt_1=np.roll(dt,1)
    diffarr=dt_1-dt
    print("Average diff: " ,np.mean(diffarr))
    print("Diff Close ratio" ,abs(np.mean(diffarr)/np.mean(dt)))
    # plt.figure(coinstat.counter )
    # plt.plot(dt)
    # plt.ylabel(coin)
    # plt.show(block=False)
    # plt.show()
coinstat.counter = 0


def createCoinData(coin,startTime,endTime):
    lastEma = 0
    historicalData = {}
    movingPeriod = 100
    ticker=[]


    historicalData=getHist (coin,startTime,endTime)

    df=pa.DataFrame(historicalData[coin])
    df.to_csv('./data_org/' + coin + '.csv', sep='\t', encoding='utf-8')

    f = open('./data/' + coin + '.csv', 'w')

    closedata = pa.DataFrame(historicalData[coin])
    historicalDataPA = closedata['close'].as_matrix()
    alldataPA= closedata.as_matrix()

    coinstat(coin, historicalDataPA)
    #print(alldataPA[1,:])
    #print(historicalData[coin][1])

    HistoricalDataOrderPandas = ["close","date","high","low","open","quoteVolume","volume","weightedAverage"] 

    # for row in historicalData[coin]:
    for i in range (0,len(historicalData[coin])):
        row = historicalData[coin][i]
        ticker.append(row['close'])
        if (len(ticker)== movingPeriod):
            ticker.pop(0)
            # write row to target file
            f.write("%.12f,%.12f,%.12f,%.12f,%.12f,%.12f," % (\
                float("{0:.12f}".format(row['volume'])),
                float("{0:.12f}".format(row['open'])),
                float("{0:.12f}".format(row['close'])),
                float("{0:.12f}".format(row['high'])),
                float("{0:.12f}".format(row['low'])),
                float("{0:.12f}".format(row['weightedAverage']))))
            # process the moving features and write them to the target file
            lastEma = createFeatures (f,ticker,movingPeriod,lastEma)
            # create Y - try to sell the coin with profit if bought now (1 - success / 0 - fail)
            

            y = testSellOpportunityEX (historicalData[coin],i,historicalDataPA)

            f.write("%.12f,\n" % y)
    f.close()


def expandtestdatabyones(X_train, y_train):
    num_ones= np.sum(y_train)
    Y_train_1=y_train.reshape(y_train.shape[0],1)
    testdata = np.append(X_train,Y_train_1,axis=1)
    print(X_train.shape)
    print(testdata.shape)
    testdata = testdata[(-testdata[:,158]).argsort()]
    ones_testdata=testdata[0:int(num_ones-1),:]
    print(ones_testdata.shape)
    for i in xrange(1):
        ones_testdata= np.append(ones_testdata,ones_testdata,axis=0)
    aa=(np.random.rand(ones_testdata.shape[0],ones_testdata.shape[1]) -0.5)
    diffmat= np.multiply(aa,ones_testdata)*0.0000
    ones_testdata=ones_testdata+diffmat
    ones_testdata[:,ones_testdata.shape[1]-1]=1

    print(ones_testdata.shape)
    return ones_testdata



def createModel(coin):

    # load data
    dataset = loadtxt('./data/'+coin+'.csv', delimiter=",", usecols=range(0, 160))
    # split data into X and y
    X = dataset[:, 0:158]
    Y = dataset[:, 159]
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # expanded = expandtestdatabyones(X_train, y_train)

    # X_train=np.append(X_train,expanded[:,0:158],axis=0)
    # y_train=np.append(y_train,expanded[:,158],axis=0)

    num_ones= np.sum(y_train)

    
   # expandtestdatabyones()

    # fit model on training data
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, y_pred)
    print cm

    # save model
    pickle.dump(model, open("./model/"+coin+".dat", "wb"))

def main(argv):
    import timeit
    start_global_time = timeit.default_timer()
    coins1 = ['BTC_ETH' ,'BTC_AMP' ,'BTC_ARDR', 'BTC_BCN','BTC_BCY','BTC_BELA','BTC_BLK', 'BTC_BTCD','BTC_BTM','BTC_BTM','BTC_BURST','BTC_CLAM','BTC_DASH','BTC_DCR','BTC_DGB','BTC_DOGE','BTC_EMC2','BTC_ETC','BTC_ETH','BTC_EXP','BTC_FCT','BTC_FLDC','BTC_FLO','BTC_GAME','BTC_GNO']
    coins2 = ['BTC_ETC','BTC_ETH','BTC_EXP','BTC_FCT','BTC_FLDC','BTC_FLO','BTC_GAME','BTC_GNO', 'BTC_GNT','BTC_GRC','BTC_HUC','BTC_LBC','BTC_LSK','BTC_LTC','BTC_MAID','BTC_NAUT','BTC_NAV','BTC_NEOS','BTC_NMC','BTC_NOTE','BTC_NXC']
    coins3 =['BTC_NXT','BTC_PASC','BTC_PINK','BTC_POT','BTC_PPC','BTC_RADS','BTC_REP','BTC_RIC','BTC_SBD','BTC_SC','BTC_SJCX','BTC_STEEM','BTC_STRAT','BTC_STR','BTC_SYS','BTC_VIA','BTC_VRC','BTC_VTC','BTC_XBC','BTC_XCP','BTC_XEM','BTC_XMR']

    coins = coins1 #['BTC_BCN']

    for coin in coins:
        start = timeit.default_timer()
        startTime = calendar.timegm(time.strptime('01/04/2017', '%d/%m/%Y'))
        endTime = calendar.timegm(time.strptime('30/06/2017', '%d/%m/%Y'))

        # create csv data file with all features and Y
        createCoinData(coin,startTime,endTime)
        # create xgboost model and save it to a file
        createModel(coin)
        elapsed = timeit.default_timer() - start
        print("Elepased Time   for ", coin,"    :",elapsed)
    print("Elapsed time for all :", start_global_time-timeit.default_timer())

if __name__ == "__main__":
    main(sys.argv[1:])
