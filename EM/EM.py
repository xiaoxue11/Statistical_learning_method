# -*- coding: utf-8 -*-
"""
Created on Thu May 30 23:14:41 2019

@author: 29132
"""

import numpy as np

class EM:
    def __init__(self,prob,max_iter):
        self.pro_A,self.pro_B,self.pro_C=prob
        self.max_iter=max_iter
    #E_step
    
    def calculate_mean(self,label):
        pro_1=self.pro_A*np.power(self.pro_B,label)*np.power(1-self.pro_B,(1-label))
        pro_2=self.pro_A*np.power(self.pro_C,label)*np.power(1-self.pro_C,(1-label))
        mu=pro_1/(pro_2+pro_1)
        return mu
    
    def fit(self,labels):
        m=len(labels)
        print('Initial pro is {},{},{}'.format(self.pro_A,self.pro_B,self.pro_C))
        for i in range(self.max_iter):
            mu=[self.calculate_mean(label) for label in labels]
            temp=[1]*m
            _mu=[i-j for i,j in zip(temp,mu)]
            pro_A=sum(mu)/m
            pro_B=(sum([i*j for i,j in zip(mu,labels)]))/sum(mu)
            pro_C=(sum([i*j for i,j in zip(_mu,labels)]))/sum(_mu)
            print('After {} iteration,the value is:{},{},{}'.format(i,pro_A,pro_B,pro_C))
            self.pro_A=pro_A
            self.pro_B=pro_B
            self.pro_C=pro_C
            

labels=[1,1,0,1,0,0,1,0,1,1] 
prob1=[0.5,0.5,0.5]
em=EM(prob=prob1,max_iter=2)
em.fit(labels)       
       
prob2=[0.4,0.6,0.7]
em=EM(prob=prob2,max_iter=2)
em.fit(labels)            
            
        