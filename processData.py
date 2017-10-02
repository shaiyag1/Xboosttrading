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

def testSellOpportunity (bid,historicalData,i):
    upMarkDiff = 1
    upMark = 0.0
    stopLoss1Trigger = False
    sellThresh = 0.99
    stopLoss1 = 1.01
    stopLoss2 = 0.90

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
            return (newPrice/bid > 1.05)
    return 0 # in case we get here


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


def createFeatures (fname,tickerclose, tickerVolume, ticketWA, movingPeriod,lastEma):


    bclose = np.array(tickerclose)
    bcloseOneprev= np.roll(bclose,1)
    bclose5prev= np.roll(bclose,5)
    bonediff=1*(bclose>bcloseOneprev)

    fname.write ("%.12f," % bcloseOneprev[0]   )
    fname.write ("%.12f," % bclose5prev[0])
    fname.write ("%.12f," % np.sum(bonediff[1:10] ))
    fname.write ("%.12f," % np.sum(bonediff[1:30]))

    btickerVolume =np.array(tickerVolume)
    btickerVolumeOneprev= np.roll(btickerVolume,1)
    btickerVolume5prev= np.roll(btickerVolume,5)
    btickerVolumeOneDiff=1*(btickerVolume>btickerVolumeOneprev)
    fname.write ("%.12f," % bcloseOneprev[0])
    fname.write ("%.12f," % bclose5prev[0])
    fname.write ("%.12f," % np.sum(btickerVolumeOneDiff[1:10]))
    fname.write ("%.12f," % np.sum(btickerVolumeOneDiff[1:30]))

# Volume * Close
    fname.write ("%.12f," % (np.sum(btickerVolumeOneDiff[1:30])-np.sum(btickerVolumeOneDiff[30:60]) ))
    fname.write ("%.12f," % np.mean(bclose[1:30])-np.sum(bclose[30:60]))


    rolling_max = max(bclose[:-1])
    fname.write ("%.12f," % rolling_max)
    rolling_max = max(bclose[:-2])
    fname.write ("%.12f," % rolling_max)
    rolling_max = max(bclose[-50:-1])
    fname.write ("%.12f," % rolling_max)
    rolling_max = max(bclose[-50:-2])
    fname.write ("%.12f," % rolling_max)
    rolling_max = max(bclose[-75:-1])
    fname.write ("%.12f," % rolling_max)
    rolling_max = max(bclose[-75:-2])
    fname.write ("%.12f," % rolling_max)






    for i in range(0,-60, -5):
        rolling_mean = float("{0:.12f}".format(np.mean(bclose[i:])))
        fname.write("%.12f," % rolling_mean)
        rolling_std = float("{0:.12f}".format(np.std(bclose[i:], ddof=0)))
        fname.write ("%.12f," % rolling_std)
        for j in 0.382,1,2,3,4:
            upper_band = float("{0:.12f}".format(rolling_mean + (rolling_std * j)))
            fname.write("%.12f," % upper_band)
            lower_band = float("{0:.12f}".format(rolling_mean - (rolling_std * j)))
            fname.write("%.12f," % lower_band)

    #EMA
    if lastEma == 0:
        ema = float("{0:.12f}".format(np.mean(bclose)))
        fname.write("%.12f," % ema)
        fname.write("%.12f," % ema)
        fname.write("%.12f," % ema)
    else:
        ema = (float(tickerclose[-1]) * 2 / (5 + 1) + lastEma * (1 - 2 / (5 + 1)))
        fname.write("%.12f," % ema)
        ema = (float(tickerclose[-1]) * 2 / (10 + 1) + lastEma * (1 - 2 / (10 + 1)))
        fname.write("%.12f," % ema)
        ema = (float(tickerclose[-1]) * 2 / (20 + 1) + lastEma * (1 - 2 / (20 + 1)))
        fname.write("%.12f," % ema)

    return ema

def createCoinData(coin,startTime,endTime):
    lastEma = 0
    historicalData = {}
    movingPeriod = 100
    tickerclose=[]
    tickerVolume=[]
    ticketWA=[]


    rrr=10
    print(type(tickerclose))

    historicalData=getHist (coin,startTime,endTime)
    fname = open('./data/' + coin + '.csv', 'w')

    # for row in historicalData[coin]:
    for i in range (0,len(historicalData[coin])):
        row = historicalData[coin][i]

        tickerclose.append(row['close'])
        tickerVolume.append(row['volume'])
        ticketWA.append(row['weightedAverage'])
        # print(tickerclose)
        # input('ssss')
        if (len(tickerclose)== movingPeriod):
            # if(rrr>0):
            #     print(row)
            #     print(tickerclose.pop(0))
            #     print(tickerclose)
            #     print(i)
            #     rrr=rrr-1
            tickerclose.pop(0)
            tickerVolume.pop(0)
            ticketWA.pop(0)

            # write row to target file
            fname.write("%.12f,%.12f,%.12f,%.12f,%.12f,%.12f," % (\
                float("{0:.12f}".format(row['volume'])),
                float("{0:.12f}".format(row['open'])),
                float("{0:.12f}".format(row['close'])),
                float("{0:.12f}".format(row['high'])),
                float("{0:.12f}".format(row['low'])),
                float("{0:.12f}".format(row['weightedAverage']))))
            # process the moving features and write them to the target file
            lastEma = createFeatures (fname,tickerclose,tickerVolume, ticketWA,movingPeriod,lastEma)
            # create Y - try to sell the coin with profit if bought now (1 - success / 0 - fail)
            y = testSellOpportunity (float(row['close']),historicalData[coin],i)
            fname.write("%.12f,\n" % y)
    fname.close()

def createModel(coin):

    # load data
    dataset = loadtxt('./data/'+coin+'.csv', delimiter=",", usecols=range(0, 170))
    # split data into X and y
    X = dataset[:, 0:45]
    Y = dataset[:, 169]
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
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
    coin = 'BTC_ETH'
    startTime = calendar.timegm(time.strptime('01/04/2017', '%d/%m/%Y'))
    endTime = calendar.timegm(time.strptime('30/06/2017', '%d/%m/%Y'))

    # create csv data file with all features and Y
    createCoinData(coin,startTime,endTime)
    # create xgboost model and save it to a file
    createModel(coin)

if __name__ == "__main__":
    main(sys.argv[1:])
