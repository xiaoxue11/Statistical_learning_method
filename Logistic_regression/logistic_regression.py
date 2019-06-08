# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:18:52 2019

@author: 29132
"""

import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('testSet.txt',delimiter='\t')
X=data[:,0:2]
y=data[:,2]

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def Gradient_descent(X,y):
    y=y.reshape(-1,1)
    m,n=X.shape
    X=np.c_[np.ones(m),X]
    alpha=0.001
    num_iter=500
    W=np.ones([n+1,1])
    for i in range(num_iter):
        h=sigmoid(np.dot(X,W))
        error=y-h
        W=W+alpha*np.dot(X.T,error)
    return W


def plot_data(X,y):
    positive=np.where(y==1)
    negtive=np.where(y==0)
    plt.figure()
    plt.plot(X[positive][:,0],X[positive][:,1],'b+')
    plt.plot(X[negtive][:,0],X[negtive][:,1],'ro')
    xarray=np.arange(-3.0, 3.0, 0.1)
    W1=Gradient_descent(X,y)
    y1=-(W1[0]+W1[1]*xarray)/W1[2]
    plt.plot(xarray,y1,'k-')
    W2=stocGradAscent0(X,y)
    y2=-(W2[0]+W2[1]*xarray)/W2[2]
    plt.plot(xarray,y2,'y-')
    plt.legend(['X1', 'X2', 'bondary1','bondary2'],loc='upper left')


def stocGradAscent0(X,y):
    y=y.reshape(-1,1)
    m,n=X.shape
    X=np.c_[np.ones(m),X]
    alpha=0.001
    W=np.ones([n+1,1])
    for i in range(m):
        h=sigmoid(np.dot(X,W))
        error=y-h
        W=W+alpha*np.dot(X.T,error)
    return W
plot_data(X,y)

    
    
