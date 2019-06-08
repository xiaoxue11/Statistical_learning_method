# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:10:19 2019

@author: 29132
"""
import numpy as np
#create data
def createDataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def knn_classfier(X,group,labels,k):
    m,n=group.shape
    distances=np.sum((np.tile(X,(m,1))-group)**2,axis=1)
    sort_indices=np.argsort(distances)
    classcount={}
    for i in range(k):
        votelabel=labels[sort_indices[i]]
        classcount[votelabel]=classcount.get(votelabel,0)+1
    sortclasscount=sorted(classcount.items(), key=lambda x:x[1], reverse=True)
    return sortclasscount[0][0]

def file2matrix(filename):
    with open(filename) as f:
        m=len(f.readlines())
        X=np.zeros([m,3])
    with open(filename) as f:
        y=[]
        index=0
        for line in f.readlines():
            line=line.strip()
            line=line.split('\t')
            X[index]=line[0:3]
            y.append(line[-1])
            index+=1
    return X,y   

    





        
    
        
        