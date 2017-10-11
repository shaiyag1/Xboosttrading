from __future__ import division
import time
import sys
import os.path
import numpy as np
from poloniex import poloniex
import pickle
from multiprocessing import Process,Pipe

sys.path.insert(0, '..')

from createfeaturesOnTop import createfeaturesOnTop

def getCandle(conn,coin,interval):
    return conn.api_query("returnChartData",
                          {"currencyPair": coin, "start": time.time() - interval,
                           "end": time.time(),"period": 300})
def getCandleP(child_conn,coin,interval):
    try:
        l=[]
        conn = poloniex('', '')
        l=conn.api_query("returnChartData",
                              {"currencyPair": coin, "start": time.time() - interval,
                               "end": time.time(),"period": 300})
        child_conn.send(l[0])
    except:
        pass

def getTickerP (child_conn):
    try:
        conn = poloniex('', '')
        t=conn.api_query('returnTicker')
        child_conn.send(t)
    except:
        pass


def main(argv):
    commInterval = 30  # 30
    samplingInterval = 300
    smplCount = samplingInterval/commInterval-1
    movingPeriod = 100  # 20
    stopLoss1 = 1.03
    stopLoss1Trigger = {}
    stopLoss2 = 0.60
    upMark = {}
    upMarkDiff =1
    sellThresh = 0.995
    newBTCTicker = {}
    movingBTCTicker = [{} for x in range(movingPeriod)]
    myOrders = {}
    diff = 0.0
    buyTrigger = False
    stop = False
    balance = 1
    wallet = 1
    moneyOut = 0
    totalInvestment = 0
    totalSales = 0
    bid = balance/2
    i = 0
    model={}
    fnames=[]
    prtCount=0




    conn = poloniex('', '')

    fnames = os.listdir("./model")
    fnames = [t for t in fnames if t.find("csv")>2]
    dataDictionaryMapping = {"close":0,"date":1,"high":2,"low":3,"open":4,"quoteVolume":5,"volume":6,"weightedAverage":7}
    listitems=["close","date","high","low","open","quoteVolume","volume","weightedAverage"]

    workingsets={}
    workingline=np.zeros((1,8))

    for fname in fnames:
        tmpworkingset=np.zeros((100,200))
        key = fname.split('.')[0]
        workingsets.update({key:tmpworkingset})

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
                currLine=l[i]              
                for itemi in xrange(len(listitems)):
                    workingline[0,itemi]=currLine[listitems[itemi]]
                    workingsets[key],lastcolom=createfeaturesOnTop(workingsets[key],workingline[0,:])

                movingBTCTicker[i][key] = l[i]
            print "%-12s Initialization " % key
        except:
            print "failed to initialize %s len is %f" % (key,len(l))
            return 0

    print "finished initialization successfuly"

    t = open('./data/transactions.csv', 'w')
    t.write ('Time, Token, Value, Trns, Amount, Profit, Balance, Max value\n')
    prt = open ('./data/protfolio_stats.csv','w')
    prt.write ('Time, Money out, Current Protfolio Value,	Protfolio ROI,	Total Investment,	Total Sales,	Total Assets, Total ROI\n')
    prt.close()


    while (not stop):

        # buying loop


        if len(movingBTCTicker) == movingPeriod:
            #print "in buying loop"
            for key in movingBTCTicker[-1]:
                # create feature list
                row = movingBTCTicker[-1][key]
                if float(row['volume'])==0:
                    #print 'skipping '+key
                    continue

                currLine=row             
                for itemi in xrange(len(listitems)):
                    workingline[0,itemi]=currLine[listitems[itemi]]
                    workingsets[key],lastcolom=createfeaturesOnTop(workingsets[key],workingline[0,:])

                y_pred=model[key].predict(workingsets[key][0:1,0:lastcolom-1])

                # check prediction
                buyTrigger = False
                if y_pred[0] == 1:
                    buyTrigger = True

                if buyTrigger and (not (key in myOrders.keys())):# and (balance >= bid):
                    orderNumber = {}
                    wallet -= bid
                    moneyOut -= bid
                    totalInvestment -= bid
                    balance -= bid
                    myOrders[key] = [movingBTCTicker[-1][key]['close'], orderNumber, 0]
                    ctime=time.strftime("%d %b %Y %H:%M:%S", time.localtime())
                    print "==> %s buying %-10s at %.8f" % (ctime, key, float(movingBTCTicker[-1][key]['close']))
                    #Time, Token, Value, Trns, Amount, Balance
                    t = open('./data/transactions.csv', 'a')
                    t.write("%s,%-12s,%10.8f,Buy,%10f,,%10f\n" % (ctime, key, float(movingBTCTicker[-1][key]['close']),-bid,balance))
                    t.close()

        # selling loop
        ctime = time.strftime("%d %b %Y %H:%M:%S", time.localtime())
        #print "before==> %s" % ctime
        while True:
            parent_conn, child_conn = Pipe()
            p = Process(target=getTickerP, args=(child_conn,))
            p.start()
            if parent_conn.poll(5):
                ticker = parent_conn.recv()
                #print "successfully read ticker"
                break
            else:
                print 'Connection timeout in getTickerP - Retrying'
                p.terminate()
                p.join()

        #ticker = conn.api_query('returnTicker')
        #print "after"
        for key in ticker:
            if (key.startswith("BTC")):
                newBTCTicker[key] = float(ticker[key]['last'])

        tmpOrders = {}
        tmpOrders = myOrders.copy()
        protfolioValue = 0.0
        for key in tmpOrders:

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
                protfolioValue += diff*bid
            else:
                diff = 0

            if diff > stopLoss1:
                if key not in stopLoss1Trigger.keys():
                    stopLoss1Trigger[key]=True
            
            maxValue = float(upMark[key])
            print "%-10s currently at %.12f purchasded at %.12f diff= %.2f current max value=%.12f" % (key,float(newBTCTicker[key]),float(tmpOrders[key][0]),float(newBTCTicker[key])/float(tmpOrders[key][0]),maxValue)

            if (((upMarkDiff <= sellThresh or diff<stopLoss1) and key in stopLoss1Trigger.keys()) or diff < stopLoss2):
                if key in stopLoss1Trigger.keys():
                    del stopLoss1Trigger[key]
                if key in upMark.keys():
                    del upMark[key]

                wallet += bid
                moneyOut += bid
                balance += (bid * diff)
                totalSales += bid * diff

                ctime = time.strftime("%d %b %Y %H:%M:%S", time.localtime())
                print "<== %s selling %-12s at %s with ROI of %s wallet: %s balance: %s" % (ctime, key, newBTCTicker[key], diff, wallet,balance)
                # Time, Token, Value, Trns, Amount, Balance
                t = open('./data/transactions.csv', 'a')
                t.write("%20s, %-10s, %.12f,Sell ,%10f,%-14f,  %-14f,  %.12f  \n" % (ctime, key, float(newBTCTicker[key]), bid*diff, diff, balance, maxValue/(float(newBTCTicker[key])/diff)))
                t.close()
                del myOrders[key]


        time.sleep(int(commInterval))
        
        prtCount += 1
        if prtCount*commInterval >= 3600:
            prtCount = 0
            prt = open('./data/protfolio_stats.csv','a')
            ctime = time.strftime("%d %b %Y %H:%M:%S", time.localtime())
            prt.write("%s, %.2f, %.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (ctime,moneyOut,protfolioValue,protfolioValue/-moneyOut,totalInvestment,totalSales,totalSales+protfolioValue,(totalSales+protfolioValue)/-totalInvestment))
            prt.close()

        smplCount += 1
        if smplCount == samplingInterval/commInterval:
            smplCount = 0
            movingBTCTicker.pop(0)
            movingBTCTicker.append(movingBTCTicker[-1])
            l=[]
            for key in movingBTCTicker[-1]:
                while True:
                    parent_conn, child_conn = Pipe()
                    p = Process(target=getCandleP, args=(child_conn,key, samplingInterval))
                    p.start()
                    if parent_conn.poll(5):
                        movingBTCTicker[-1][key]= parent_conn.recv()
                        #print "successfully read candle for %s" % key
                        #print movingBTCTicker[-1][key]
                        break
                    else:
                        print 'Connection timeout in getCandleP - Retrying'
                        p.terminate()
                        p.join()
                    #movingBTCTicker[-1][key] = getCandle(conn, key, samplingInterval)[0]

if __name__ == "__main__":
    main(sys.argv[1:])
