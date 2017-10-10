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

from createfeaturesOnTop import createfeaturesOnTop



def testSellOpportunityEX (sampleIndex,historicalDataPA):
 
    sellThresh = 0.99
    ROItresh  =1.09
    futurerange=12*4

    bid = historicalDataPA[sampleIndex]
    if bid == 0:
        return 0

    #closedata = pa.DataFrame(historicalData)
    #closedataarray = closedata['close'].as_matrix()
    till =min(sampleIndex+futurerange,np.size(historicalDataPA))
    if(till<0.5*futurerange):
        return 0
    max_index_arr=np.argmax(historicalDataPA[sampleIndex:till])
    max_index_arr=max_index_arr+sampleIndex
    if(np.max(historicalDataPA[sampleIndex:till])<ROItresh*bid):
        return 0
    if(np.max(historicalDataPA[sampleIndex:till])!=historicalDataPA[max_index_arr]):
        for j in xrange(sampleIndex,till):
            if(np.max(historicalDataPA[sampleIndex:till])== historicalDataPA[j]):
                print(" Max index ", j ,historicalDataPA[j] )
        print('  %d   %10f     %10f    %10f ',sampleIndex,bid, max_index_arr, np.max(historicalDataPA[sampleIndex:till]),historicalDataPA[max_index_arr])
        input('Wrong Assumption')


    # Not accurate, we should find thar drastic min doesn't come befor the peak

    if(np.size(max_index_arr)==1):
        min_index_arr = np.argmin(historicalDataPA[sampleIndex:till])+sampleIndex
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

    return historicalData




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


def createCoinDataEX(coin,startTime,endTime,mode='offline'):
        lastEma = 0
        historicalData = {}
        movingPeriod = 200
        ticker=[]


        if('offline' == mode):
            print('offline mode')
            tradedata= pa.read_csv('./data_org2/' + coin + '.csv', sep='\t', encoding='utf-8')
        else:
            historicalData=getHist (coin,startTime,endTime)
            df=pa.DataFrame(historicalData[coin])
            df.to_csv('./data_org2/' + coin + '.csv', sep='\t', encoding='utf-8')
            tradedata = pa.DataFrame(historicalData[coin])


        tradedataMA=tradedata.as_matrix()
        if('offline' == mode):
            tradedataMA=tradedataMA[:,1:tradedataMA.shape[1]]



            
        historicalDataPA = tradedata['close'].as_matrix()
        alldataPA= tradedata.as_matrix()
        retdataset=np.copy(alldataPA[movingPeriod:alldataPA.shape[0],:])
        retdataset =np.append(retdataset, np.zeros((retdataset.shape[0],200)),axis=1)
        retmaxindex=0
        HistoricalDataOrderPandas = {"close":0,"date":1,"high":2,"low":3,"open":4,"quoteVolume":5,"volume":6,"weightedAverage":7}

        #coinstat(coin, tradedataMA[:,HistoricalDataOrderPandas['close']])

        # Initialize working set

        workingset= np.zeros((movingPeriod,retdataset.shape[1]))
        for i in xrange(movingPeriod):
            workingset[movingPeriod-i-1,0:8]=tradedataMA[i,0:8]



        for   i in xrange (retdataset.shape[0]):

            ioffset=i+movingPeriod
            j=8

            if(True):
                workingset,retmaxindex= createfeaturesOnTop(workingset,tradedataMA[ioffset,:])
                workingset[0,retmaxindex]=testSellOpportunityEX (ioffset,tradedataMA[:,HistoricalDataOrderPandas['close'] ]  )
                retdataset[i,:]=workingset[0,:]

            else:
                for item in ["quoteVolume","volume","weightedAverage",'close']:
                    for k in [2,5,10,20,40,70]:
                        retdataset[i,j]=np.max(tradedataMA[ioffset-k:ioffset,HistoricalDataOrderPandas[item]])
                        j=j+1
                        retdataset[i,j]=np.min(tradedataMA[ioffset-k:ioffset,HistoricalDataOrderPandas[item]])
                        j=j+1
                        retdataset[i,j]=np.mean(tradedataMA[ioffset-k:ioffset,HistoricalDataOrderPandas[item]])
                        j=j+1
                        retdataset[i,j]=np.min(tradedataMA[ioffset-k:ioffset,HistoricalDataOrderPandas[item]])-tradedataMA[ioffset,HistoricalDataOrderPandas[item]]
                        j=j+1
                        retdataset[i,j]=tradedataMA[ioffset][HistoricalDataOrderPandas[item]]-tradedataMA[i-k][HistoricalDataOrderPandas[item]]
                        j=j+1
                y = testSellOpportunityEX (ioffset,tradedataMA[:,HistoricalDataOrderPandas['close'] ]  )
                retdataset[i,j]=y
                retmaxindex=j


        return retdataset,retmaxindex


def expandtestdatabyones(X_train, y_train):
    num_ones= np.sum(y_train)
    Y_train_1=y_train.reshape(y_train.shape[0],1)
    testdata = np.append(X_train,Y_train_1,axis=1)
    # print(X_train.shape)
    # print(testdata.shape)
    testdata = testdata[(-testdata[:,158]).argsort()]
    ones_testdata=testdata[0:int(num_ones-1),:]
    # print(ones_testdata.shape)
    for i in xrange(1):
        ones_testdata= np.append(ones_testdata,ones_testdata,axis=0)
    aa=(np.random.rand(ones_testdata.shape[0],ones_testdata.shape[1]) -0.5)
    diffmat= np.multiply(aa,ones_testdata)*0.0000
    ones_testdata=ones_testdata+diffmat
    ones_testdata[:,ones_testdata.shape[1]-1]=1

    # print(ones_testdata.shape)
    return ones_testdata




def createModelEX(coin,datasetMA,naxcolom):

    # load data

    # split data into X and y

    X = datasetMA[:, 0:naxcolom-1]
    Y = datasetMA[:, naxcolom]   # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # expanded = expandtestdatabyones(X_train, y_train)

    num_ones= np.sum(y_train)

    
   # expandtestdatabyones()

    # fit model on training data
    model = XGBClassifier()
    model.fit(X_train, y_train)


    # make prediction to train data 

    y_pred= model.predict(X_train)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(predictions, y_train)
    cm = confusion_matrix(y_train, y_pred)
    print("Training set results:")
    print cm



    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=4)
    print("Test set results:")
    print cm
    #print(model.feature_importances_)
    # save model
    pickle.dump(model, open("./model/"+coin+".dat", "wb"))
    return cm

def main(argv):
    import timeit
    start_global_time = timeit.default_timer()
    coins1_partial = ['BTC_ETH' ,'BTC_AMP' ,'BTC_ARDR', 'BTC_BCN','BTC_BCY','BTC_BELA','BTC_BLK']
    coins1 = ['BTC_ETH' ,'BTC_AMP' ,'BTC_ARDR', 'BTC_BCN','BTC_BCY','BTC_BELA','BTC_BLK', 'BTC_BTCD','BTC_BTM','BTC_BTM','BTC_BURST','BTC_CLAM','BTC_DASH','BTC_DCR','BTC_DGB','BTC_DOGE','BTC_EMC2','BTC_ETC','BTC_ETH','BTC_EXP','BTC_FCT','BTC_FLDC','BTC_FLO','BTC_GAME','BTC_GNO']
    coins2 = ['BTC_ETC','BTC_ETH','BTC_EXP','BTC_FCT','BTC_FLDC','BTC_FLO','BTC_GAME','BTC_GNO', 'BTC_GNT','BTC_GRC','BTC_HUC','BTC_LBC','BTC_LSK','BTC_LTC','BTC_MAID','BTC_NAUT','BTC_NAV','BTC_NEOS','BTC_NMC','BTC_NOTE','BTC_NXC']
    coins3 =['BTC_NXT','BTC_PASC','BTC_PINK','BTC_POT','BTC_PPC','BTC_RADS','BTC_REP','BTC_RIC','BTC_SBD','BTC_SC','BTC_SJCX','BTC_STEEM','BTC_STRAT','BTC_STR','BTC_SYS','BTC_VIA','BTC_VRC','BTC_VTC','BTC_XBC','BTC_XCP','BTC_XEM','BTC_XMR']

    coins = coins1 #coins1_partial #coins1+coins2+ coins3#['BTC_BCN']
    cm =[[0,0],[0,0]]
    for coin in coins:
        start = timeit.default_timer()
        startTime = calendar.timegm(time.strptime('01/04/2017', '%d/%m/%Y'))
        endTime = calendar.timegm(time.strptime('30/06/2017', '%d/%m/%Y'))

        # create csv data file with all features and Y
        dataset,naxcolom = createCoinDataEX(coin,startTime,endTime)
        # create xgboost model and save it to a file
        cm=cm+createModelEX(coin,dataset,naxcolom)
        #createModel(coin)


        elapsed = timeit.default_timer() - start
        print("Elepased Time   for ", coin,"    :",elapsed)
    print("Sum CM")
    print(cm)
    print(cm/np.sum(cm))
    print("Elapsed time for all :", start_global_time-timeit.default_timer())

if __name__ == "__main__":
    main(sys.argv[1:])
