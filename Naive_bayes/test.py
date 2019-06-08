# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:19:48 2019

@author: 29132
"""

a=[0]*5
print(a.index(0))

y=[-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
m=y.count(1)

x=[1,2,3,4,5,6,8]
x_sum=sum(x)
x_mean=sum(x)/len(x)
averg=sum([(i-x_mean)**2 for i in x])

def mean(X):
    return sum(X)/len(X)

def std(X):
    x_mean=mean(X)
    return sum([(i-x_mean)**2 for i in X])

X=[[1,2,3,4,5,6],[7,8,9,10,11,12]]
y=[1,2]

def getparams(X):
    summary=[(mean(data),std(data)) for data in zip(X)]
    return summary

labels=set(y)
data={label:[] for label in labels}
for f,label in zip(X,y):
    data[label].extend(f)
model={label:getparams(value) for label,value in data.items()}
