
import numpy as np

def createfeaturesOnTop(workingset,tickerline):
    j=8
    workingset=np.roll(workingset,1,axis=0)

    itemorder = {"close":0,"date":1,"high":2,"low":3,"open":4,"quoteVolume":5,"volume":6,"weightedAverage":7}

    workingset[0,0:j] = tickerline[0:j]

    for item in ["quoteVolume","volume","weightedAverage",'close']:
        for k in [2,5,10,20,40]:
            workingset[0,j]=np.max(workingset[0:k,itemorder[item]])
            j=j+1
            workingset[0,j]=np.min(workingset[0:k,itemorder[item]])
            j=j+1
            workingset[0,j]=np.mean(workingset[0:k,itemorder[item]])
            j=j+1
            workingset[0,j]=np.min(workingset[0:k,itemorder[item]])-workingset[0,itemorder[item]]
            j=j+1
            workingset[0,j]=workingset[0,itemorder[item]]-workingset[k][itemorder[item]]
            j=j+1
    numcoloms=j

    return workingset,numcoloms