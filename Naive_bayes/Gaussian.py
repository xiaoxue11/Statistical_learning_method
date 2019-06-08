# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:15:22 2019

@author: 29132
"""

import math
class Gaussian_NB:
    def __init__(self):
        self.model=None
        
    @staticmethod
    def mean(X):
        return sum(X)/len(X)
    
    def std(self,X):
        average=self.mean(X)
        return math.sqrt(sum([(i-average)**2 for i in X])/(len(X)-1))

    #===split train data by ylabel
    def getparams(self,X):
        summary=[(self.mean(data),self.std(data)) for data in zip(*X)]
        return summary
    
    def guassian_probablity(self,x,mean,std):
        exponent=(x-mean)**2/(2*std*std)
        probablity=math.exp(-exponent)/(2*math.pi*std*std)**0.5
        return probablity
    
    def fit(self,X,y):
        labels=set(y)
        data={label:[] for label in labels}
        for f,label in zip(X,y):
            data[label].append(f)
        self.model={label:self.getparams(value) for label,value in data.items()}
        print('Guassian model trained')
    
    def cal_guassian_pro(self,input_data):
        probably={}
        for label,value in self.model.items():
            probably[label]=1.0
            for i in range(len(value)):
                (mean,std)=value[i]
                probably[label]*=self.guassian_probablity(input_data[i],mean,std)
        return probably
    
    def predict(self,test):
        pred=sorted(self.cal_guassian_pro(test).items(),key=lambda x:x[1],reverse=True)
        return pred[0][0]
    
    def score(self,X_test,y_test):
        count=0
        m=len(y_test)
        for i in range(m):
            if self.predict(X_test[i])==y_test[i]:
                count+=1
        accuracy=count/m
        return accuracy
    
from sklearn import datasets
iris=datasets.load_iris()
X=iris['data']
y=iris['target']

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf=GaussianNB()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

model=Gaussian_NB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
        
    
        
        
    
    