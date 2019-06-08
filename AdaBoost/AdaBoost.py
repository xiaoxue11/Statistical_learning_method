# -*- coding: utf-8 -*-
"""
Created on Tue May 28 23:55:45 2019

@author: 29132
"""
import numpy as np
class AdaBoost:
    def __init__(self,n_estimators,learning_rate):
        self.estimator=n_estimators
        self.lr=learning_rate
        self.weakclass=[]
        
    def stumpClassify(self,data,axis,threshVal,threshIneq):
        m,n=data.shape
        retarray=np.ones([m,1])
        if threshIneq=='lt':
            retarray[data[:,axis]<threshVal]=-1.0
        else:
            retarray[data[:,axis]>threshVal]=-1.0
        return retarray
    
    def buildStump(self,data,labels,D):
        labels=labels.reshape(-1,1)
        m,n=data.shape
        min_error=np.inf
        bestStump={}
        for i in range(n):
            max_value=data[:,i].max()
            min_value=data[:,i].min()
            num_steps=(max_value-min_value)//self.lr
            for j in range(-1,int(num_steps)+1):
                for Ineq in ['lt','gt']:
                    threshValue=min_value+j*self.lr
                    predictarray=self.stumpClassify(data,i,threshValue,Ineq)
                    error=np.ones([m,1])
                    error[predictarray==labels]=0
                    weight_error=np.dot(D.T,error)
                    if weight_error<min_error:
                        min_error=weight_error
                        bestClasEst=predictarray
                        bestStump['dim']=i
                        bestStump['thresh']=threshValue
                        bestStump['ineq']=Ineq
        return bestStump,bestClasEst,min_error  
      
    def fit(self,X,y):
        y=y.reshape(-1,1)
        m,n=X.shape
        D=np.ones([m,1])/m
        aggClassEst=np.zeros([m,1])
        for i in range(self.estimator):
            bestStump,bestClasEst,error=self.buildStump(X,y,D)
            alpha=0.5*np.log((1-error)/max(error,1e-16))
            bestStump['alpha']=alpha
            self.weakclass.append(bestStump)
            expont=alpha*y*bestClasEst
            D=np.multiply(D,np.exp(-expont))
            D=D/np.sum(D)
            aggClassEst+=alpha*bestClasEst
            aggErrors = np.multiply(np.sign(aggClassEst) !=y,np.ones((m,1)))
            errorRate = aggErrors.sum()/m
            if errorRate == 0.0: 
                break
        print('AdaBoost train done!')
    
    def predict(self,X): 
        m=X.shape[0]
        aggClassEst = np.zeros([m,1])
        for i in range(len(self.weakclass)):
            classEst = self.stumpClassify(X,self.weakclass[i]['dim'],\
                                 self.weakclass[i]['thresh'],\
                                 self.weakclass[i]['ineq'])
            aggClassEst += self.weakclass[i]['alpha']*classEst
        return np.sign(aggClassEst)
    
    def score(self,X_test,y_test):
        y_pred=self.predict(X_test)
        m=X.shape[0]
        y_test=y_test.reshape(-1,1)
        count=np.ones([m,1])
        count[y_pred!=y_test]=0
        accuracy=np.sum(count)/m
        return accuracy
#%%
X=np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1,1)
y=np.array([1,1,1,-1,-1,-1,1,1,1,-1])
#%%
clf1=AdaBoost(30,1)
clf1.fit(X,y)
accuracy1=clf1.score(X,y)
#%%
from sklearn.ensemble import AdaBoostClassifier
clf2 = AdaBoostClassifier(n_estimators=100, random_state=0)
clf2.fit(X, y)  
accuracy2=clf2.score(X, y) 
  
    
    