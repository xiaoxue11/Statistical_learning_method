# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:27:21 2019

@author: 29132
"""
import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():
    datMat = np.array([[ 1. , 2.1],
                     [ 2. , 1.1],
                     [ 1.3, 1. ],
                     [ 1. , 1. ],
                     [ 2. , 1. ]])
    classLabels =np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    return datMat,classLabels
data,labels=loadSimpData()
postive=np.where(labels==1)
negtive=np.where(labels==-1)
plt.figure()
plt.plot(data[postive][:,0],data[postive][:,1],'ro')
plt.plot(data[negtive][:,0],data[negtive][:,1],'b+')

def stumpClassify(data,axis,threshVal,threshIneq):
    m,n=data.shape
    retarray=np.ones([m,1])
    if threshIneq=='lt':
        retarray[data[:,axis]<threshVal]=-1.0
    else:
        retarray[data[:,axis]>threshVal]=-1.0
    return retarray

def bulidStump(data,labels,D):
    labels=labels.reshape(-1,1)
    m,n=data.shape
    numsteps=10.0
    min_error=np.inf
    bestStump={}
    for i in range(n):
        rangeMax=data[:,i].max()
        rangeMin=data[:,i].min()
        stepsize=(rangeMax-rangeMin)/numsteps
        for j in range(-1,int(numsteps)+1):
            for inequal in ['lt','gt']:
                threshVal=rangeMin+float(j)*stepsize
                predictarray=stumpClassify(data,i,threshVal,inequal)
                error=np.ones([m,1])
                error[predictarray==labels]=0.0
                weight_error=np.dot(D.T,error)
                if weight_error<min_error:
                    min_error=weight_error
                    bestClasEst=predictarray
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,bestClasEst,min_error

def AdaBoostTrain(data,labels,num_iter=40):
    labels=labels.reshape(-1,1)
    weakclass=[]
    m,n=data.shape
    D=np.ones([m,1])/m
    aggClassEst=np.zeros([m,1])
    for i in range(num_iter):
        bestStump,bestClasEst,error=bulidStump(data,labels,D)
        print("D:",D)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakclass.append(bestStump)
        print("bestClasEst: ",bestClasEst)
        expon=np.multiply(alpha*labels,bestClasEst)
        D=np.multiply(D,np.exp(-expon))
        D=D/D.sum()
        aggClassEst += alpha*bestClasEst
        print("aggClassEst: ",aggClassEst)
        aggErrors = np.multiply(np.sign(aggClassEst) !=labels,np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate,"\n")
        if errorRate == 0.0: break
    return weakclass
classifierArray = AdaBoostTrain(data,labels,9)
                
def AdaClassify(data,classifierArr): 
    m=data.shape[0]
    aggClassEst = np.zeros([m,1])
    for i in range(len(classifierArr)):
        classEst = stumpClassify(data,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

classifierArr = AdaBoostTrain(data,labels,30)
result1=AdaClassify(np.array([[0, 0]]),classifierArr)
result2=AdaClassify(np.array([[5, 5],[0,0]]),classifierArr)

#%%
X=np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1,1)
y=np.array([1,1,1,-1,-1,-1,1,1,1,-1])
classification=AdaBoostTrain(X,y,30)