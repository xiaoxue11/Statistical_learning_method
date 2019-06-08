# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:56:41 2019

@author: 29132
"""
import numpy as np
X=np.array([[1,2],[3,4],[5,6]])
w=np.ones(2)
y=np.array([1,1,-1])
m=[]
for i in range(X.shape[0]):
    X1=X[i]
    Y1=y[i]
    y_hat=np.sum(Y1*np.dot(X1,w))
    print(y_hat)
    if y_hat<=0:
        w = w + 0.01*np.dot(X1, Y1)
        print(w)