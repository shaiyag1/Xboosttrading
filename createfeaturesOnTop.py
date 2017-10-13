
import numpy as np

def createfeaturesOnTop(workingset,tickerline):
    j=8
    workingset=np.roll(workingset,1,axis=0)

    itemorder = {"close":0,"date":1,"high":2,"low":3,"open":4,"quoteVolume":5,"volume":6,"weightedAverage":7}

    workingset[0,0:j] = tickerline[0:j]

    for item in  ["quoteVolume","volume","weightedAverage",'close']:  #["quoteVolume",'close']: #
        for k in  [10 , 30,90, 130, 150 ,190]:#[2,5,10,20,40,50]:
            workingset[0,j]=np.max(workingset[0:k,itemorder[item]])
            j=j+1
            # workingset[0,j]=np.min(workingset[0:k,itemorder[item]])
            # j=j+1
            workingset[0,j]=np.mean(workingset[0:k,itemorder[item]])
            j=j+1
            workingset[0,j]=np.min(workingset[0:k,itemorder[item]])-workingset[0,itemorder[item]]
            j=j+1
            workingset[0,j]=workingset[0,itemorder[item]]-workingset[k][itemorder[item]]
            j=j+1
    for item in ["volume"]:
        for k in [6 ,15, 50 , 90 ,130]:
            workingset[0,j] =np.sum(workingset[0:k,itemorder[item]])
            j=j+1
            workingset[0,j] =np.sum(workingset[0:k,itemorder[item]])-np.sum(workingset[k:2*k,itemorder[item]])
            j=j+1
            workingset[0,j] =np.sum(workingset[0:k,itemorder[item]])*np.sum(workingset[k:2*k,itemorder[item]])
            j=j+1
            workingset[0,j] =np.sum(workingset[0:k,itemorder[item]])-np.sum(workingset[k:2*k,itemorder["close"]])

            j=j+1
            workingset[0,j] =np.sum(workingset[0:k,itemorder[item]])*np.sum(workingset[k:2*k,itemorder["close"]])
            j=j+1
    for item in ["weightedAverage"]:
        for k in [10 , 50,90, 130, 160]:
            # workingset[0,j] =np.sum(workingset[0:k,itemorder[item]])
            # j=j+1
            # workingset[0,j] =np.sum(workingset[0:k,itemorder[item]])-np.sum(workingset[k:2*k,itemorder[item]])
            # j=j+1
            workingset[0,j] =np.std(workingset[0:k,itemorder[item]])
            j=j+1
            workingset[0,j] =np.sum(workingset[0:k,itemorder[item]])*np.sum(workingset[k:2*k,itemorder[item]])
            j=j+1
            workingset[0,j] =np.sum(workingset[0:k,itemorder[item]])-np.sum(workingset[k:2*k,itemorder["close"]])
            j=j+1
            workingset[0,j] =np.sum(workingset[0:k,itemorder[item]])*np.sum(workingset[k:2*k,itemorder["close"]])
            j=j+1








    numcoloms=j

    return workingset,numcoloms