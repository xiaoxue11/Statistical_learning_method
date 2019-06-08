# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 22:18:30 2019

@author: 29132
"""
import numpy as np
#example 4.1
def load_dataset():
    groups=[['1','S'],['1','M'],['1','M'],['1','S'],['1','S'],\
            ['2','S'],['2','M'],['2','M'],['2','L'],['2','L'],\
            ['3','L'],['3','M'],['3','M'],['3','L'],['3','L']]
    labels=[-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
    return groups,labels
            
def createFeatureList(groups):
    featureset=[]
    for group in groups:
        featureset.extend(group)
    return list(set(featureset))

def setfeatures2vec(featurelist,inputset):
    resvec=[0]*len(featurelist)
    for feature in inputset:
        if feature in featurelist:
            resvec[featurelist.index(feature)]=1
    return resvec
            
def trainNB(train_matrix,train_catogry):
    train_num=len(train_matrix)
    num_features=len(train_matrix[0])
    p1=train_catogry.count(1)/float(train_num)
    p0_num=np.ones(num_features)
    p1_num=np.ones(num_features)
    p0_demo=2.0
    p1_demo=2.0
    for i in range(train_num):
        if train_catogry[i]==1:
            p1_num+=train_matrix[i]
            p1_demo+=sum(train_matrix[i])
        else:
            p0_num+=train_matrix[i]
            p0_demo+=sum(train_matrix[i])
    p1_vec=np.log(p1_num/p1_demo)
    p0_vec=np.log(p0_num/p0_demo)
    return p0_vec,p1_vec,p1
        
def classifyNB(vec2Classify, p0vec, p1vec, pClass1):
    p1=sum(vec2Classify*p1vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0vec)+np.log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return -1
    
def testNB():
    X,y = load_dataset()
    myFeatureList =createFeatureList(X)
    train_matrix=[]
    for inputset in X:
        train_matrix.append(setfeatures2vec(myFeatureList,inputset))
    p0_vec,p1_vec,p1=trainNB(train_matrix,y)
    testEntry = ['2', 'S']
    thisDoc = setfeatures2vec(myFeatureList, testEntry)
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0_vec,p1_vec,p1))

testNB()
  
        
        
        
        
        
        
            
    


            
        
