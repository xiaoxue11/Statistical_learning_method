# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:24:46 2019

@author: 29132
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def create_dataset():
    data=np.loadtxt('testSet.txt',delimiter='\t')
    X=data[:,0:2]
    y=data[:,2]
    return X,y
X,y=create_dataset()

class Logistic_Regression:
    def __init__(self,max_iter,learning_rate):
        self.max_iter=max_iter
        self.lr=learning_rate
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
        
    def fit(self,X,y):
        m,n=X.shape
        y=y.reshape(-1,1)
        X=np.c_[np.ones(m),X]
        self.W=np.ones([n+1,1])
        for i in range(self.max_iter):
            y_pred=self.sigmoid(np.dot(X,self.W))
            grad=np.dot(X.T,y_pred-y)
            new_W=self.W-self.lr*grad
            self.W=new_W
            
    def predict(self,X):
        m=X.shape[0]
        X=np.c_[np.ones(m),X]
        y_pred=np.zeros([m,1])
        result=self.sigmoid(np.dot(X,self.W))
        for i in range(m):
            if result[i,0]>0.5:
                y_pred[i]=1
        return y_pred
    
    def score(self, X_test, y_test):
        m=X_test.shape[0]
        y_pred=self.predict(X_test)
        y_test=y_test.reshape(-1,1)
        count=0
        for i in range(m):
            if y_pred[i]==y_test[i]:
                count+=1
        return count/m

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)
y_pred1=clf.predict(X[:2, :])
accuracy1=clf.score(X, y)

lr=Logistic_Regression(max_iter=200,learning_rate=0.01)
lr.fit(X,y)
y_pred2=lr.predict(X[:2,:])
accuracy2=lr.score(X, y)

def plot_data(X,y):
    positive=np.where(y==1)
    negtive=np.where(y==0)
    plt.figure()
    plt.plot(X[positive][:,0],X[positive][:,1],'b+')
    plt.plot(X[negtive][:,0],X[negtive][:,1],'ro')
    xarray=np.arange(-3.0, 3.0, 0.1)
    lr=Logistic_Regression(max_iter=200,learning_rate=0.01)
    lr.fit(X,y)
    W1=lr.W
    y1=-(W1[0]+W1[1]*xarray)/W1[2]
    plt.plot(xarray,y1,'k-')
    plt.legend(['X1', 'X2', 'bondary1'],loc='upper left')
plot_data(X,y)