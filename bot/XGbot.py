from __future__ import division
import time
import sys, getopt, traceback
import datetime
import ast
import os.path
import calendar
import xlsxwriter
import numpy as np
from collections import defaultdict
from poloniex import poloniex
import pickle

def getCandle(conn,coin,interval):
    return conn.api_query("returnChartData",
                          {"currencyPair": coin, "start": time.time() - interval,
                           "end": time.time(),"period": 300})


def createFeatures (cData,movingBTCTicker, coin, movingPeriod,lastEma):
    ticker=[]
    for i in range (0,len(movingBTCTicker)):
        ticker.append(movingBTCTicker[i][coin]['close'])
    b = np.array(ticker)
    cData.append(max(b[:-1]))
    cData.append(max(b[:-2]))
    cData.append(max(b[-50:-1]))
    cData.append(max(b[-50:-2]))
    cData.append(max(b[-75:-1]))
    cData.append(max(b[-75:-2]))

    for i in range(0,-60, -5):
        rolling_mean = float("{0:.12f}".format(np.mean(b[i:])))
        cData.append(rolling_mean)
        rolling_std = float("{0:.12f}".format(np.std(b[i:], ddof=0)))
        cData.append(rolling_std)
        for j in 0.382,1,2,3,4:
            cData.append(float("{0:.12f}".format(rolling_mean + (rolling_std * j))))
            cData.append(float("{0:.12f}".format(rolling_mean - (rolling_std * j))))

    #EMA
    if lastEma == 0:
        ema= float("{0:.12f}".format(np.mean(b)))
        cData.append(ema)
        cData.append(ema)
        cData.append(ema)
    else:
        cData.append(float(ticker[-1]) * 2 / (5 + 1) + lastEma * (1 - 2 / (5 + 1)))
        cData.append(float(ticker[-1]) * 2 / (10 + 1) + lastEma * (1 - 2 / (10 + 1)))
        ema = (float(ticker[-1]) * 2 / (20 + 1) + lastEma * (1 - 2 / (20 + 1)))
        cData.append(ema)

    return ema


def main(argv):
    commInterval = 30  # 30
    samplingInterval = 300
    smplCount = samplingInterval/commInterval-1
    movingPeriod = 100  # 20
    stopLoss1 = 1.01
    stopLoss1Trigger = {}
    stopLoss2 = 0.90
    upMark = {}
    upMarkDiff =1
    sellThresh = 0.99
    newBTCTicker = {}
    movingBTCTicker = [{} for x in range(movingPeriod)]
    myOrders = {}
    diff = 0.0
    buyTrigger = False
    stop = False
    balance = 1
    wallet = 1
    bid = balance/2
    i = 0
    model={}
    fnames=[]
    lastEma={}

    conn = poloniex('', '')

    fnames = os.listdir("./model")

    for fname in fnames:
        key = fname.split('.')[0]
        model[key] = pickle.load(open("./model/"+fname, "rb"))

        l=[]
        l = conn.api_query("returnChartData",
                           {"currencyPair": key, "start": time.time() - 300 * (movingPeriod+30),
                            "end": time.time(),
                            "period": 300})

        try:
            for i in xrange(-1, -(movingPeriod+1), -1):
                movingBTCTicker[i][key] = l[i]
            lastEma[key]=0
            print "finished initializing %s" % key
        except:
            print "failed to initialize %s len is %f" % (key,len(l))
            return 0

    print "finished initialization successfuly"

    t = open('./data/transactions.csv', 'w')
    t.write ('Time, Token, Value, Trns, Amount, Profit, Balance\n')


    while (not stop):

        # buying loop
        if len(movingBTCTicker) == movingPeriod:
            print "in buying loop"
            for key in movingBTCTicker[-1]:
                # create feature list
                row = movingBTCTicker[-1][key]
                if float(row['volume'])==0:
                    print 'skipping '+key
                    continue

                cData = [[]]
                cData[0].append(float("{0:.12f}".format(row['volume'])))
                cData[0].append(float("{0:.12f}".format(row['open'])))
                cData[0].append(float("{0:.12f}".format(row['close'])))
                cData[0].append(float("{0:.12f}".format(row['high'])))
                cData[0].append(float("{0:.12f}".format(row['low'])))
                cData[0].append(float("{0:.12f}".format(row['weightedAverage'])))

                lastEma[key] = createFeatures(cData[0], movingBTCTicker, key, movingPeriod, lastEma[key])
                # activate model
                x_temp=np.array(cData)
                y_pred = model[key].predict(x_temp[0:1,0:158])
                # check prediction
                buyTrigger = False
                if y_pred[0] == 1:
                    buyTrigger = True

                if buyTrigger and (not (key in myOrders.keys())):# and (balance >= bid):
                    orderNumber = {}
                    wallet -= bid
                    balance -= bid
                    myOrders[key] = [movingBTCTicker[-1][key]['close'], orderNumber, 0]
                    ctime=time.strftime("%d %b %Y %H:%M:%S", time.localtime())
                    print "==> %s buying %s at %.8f" % (ctime, key, float(movingBTCTicker[-1][key]['close']))
                    #Time, Token, Value, Trns, Amount, Balance
                    t = open('./data/transactions.csv', 'a')
                    t.write("%s,%s,%.8f,Buy,%f,,%f\n" % (ctime, key, float(movingBTCTicker[-1][key]['close']),-bid,balance))
                    t.close()

        # selling loop
        ctime = time.strftime("%d %b %Y %H:%M:%S", time.localtime())
        print "before==> %s" % ctime
        try:
            ticker = conn.api_query('returnTicker')
        except:
            ticker=None;
            print " conn.api_query('returnTicker') Failed" 


        print "after"
        if(ticker!=None):
            for key in ticker:
                if (key.startswith("BTC")):
                    newBTCTicker[key] = float(ticker[key]['last'])

        tmpOrders = {}
        tmpOrders = myOrders.copy()
        for key in tmpOrders:
            print "%s at %.12f purchasded at %.12f" % (key,float(newBTCTicker[key]),float(tmpOrders[key][0]))

            if key in upMark.keys():
                if float(newBTCTicker[key]) > float(upMark[key]):
                    upMark[key] = newBTCTicker[key]
                    upMarkDiff = 1
                else:
                    upMarkDiff = float(newBTCTicker[key])/ float(upMark[key])
            else:
                if key in newBTCTicker.keys():
                    upMark[key]=newBTCTicker[key]
                    upMarkDiff =1
                else:
                    print key+' not found'

            if float(tmpOrders[key][0]) >0:
                diff = float(newBTCTicker[key]) / float(tmpOrders[key][0])
            else:
                diff = 0

            if diff > stopLoss1:
                if key not in stopLoss1Trigger.keys():
                    stopLoss1Trigger[key]=True

            if (upMarkDiff <= sellThresh or (diff<stopLoss1 and key in stopLoss1Trigger.keys()) or diff < stopLoss2):
                if key in stopLoss1Trigger.keys():
                    del stopLoss1Trigger[key]
                if key in upMark.keys():
                    del upMark[key]

                wallet += bid
                balance += (bid * diff)

                ctime = time.strftime("%d %b %Y %H:%M:%S", time.localtime())
                print "<== %s selling %s at %s with profit of %s wallet: %s balance: %s" % (ctime, key, newBTCTicker[key], diff, wallet,balance)
                # Time, Token, Value, Trns, Amount, Balance
                t = open('./data/transactions.csv', 'a')
                t.write("%s,%s,%.8f,Sell,%f,%f,%f\n" % (ctime, key, float(newBTCTicker[key]), bid*diff, diff, balance))
                t.close()
                del myOrders[key]


        time.sleep(int(commInterval))

        smplCount += 1
        if smplCount == samplingInterval/commInterval:
            smplCount = 0
            movingBTCTicker.pop(0)
            movingBTCTicker.append(movingBTCTicker[-1])
            l=[]
            for key in movingBTCTicker[-1]:
                movingBTCTicker[-1][key] = getCandle(conn, key, samplingInterval)[0]

if __name__ == "__main__":
    main(sys.argv[1:])
