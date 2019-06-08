# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 22:32:21 2019

@author: 29132
"""

import numpy as np

def createDataSet():
    dataSet = [['youth','no','no','well','no'],
               ['youth','no','no','good','no'],
               ['youth','yes','no','good','yes'],
               ['youth','yes','yes','well','yes'],
               ['youth','no','no','well','no'],
               ['senior','no','no','well','no'],
               ['senior','no','no','good','no'],
               ['senior','yes','yes','good','yes'],
               ['senior','no','yes','excellent','yes'],
               ['senior','no','yes','excellent','yes'],
               ['old','no','yes','excellent','yes'],
               ['old','no','yes','good','yes'],
               ['old','yes','no','good','yes'],
               ['old','yes','no','excellent','yes'],
               ['old','no','no','well','no']]
    labels = ['age','work','house','credit']
    return dataSet, labels

def calc_entropy(dataSet):
    m=len(dataSet)
    labelcounts={}
    for i in range(m):
        label=dataSet[i][-1]
        labelcounts[label]=labelcounts.get(label,0)+1
    entropy=0.0
    for counts in labelcounts.values():
        prob=counts/m
        entropy-=prob*np.log2(prob)
    return entropy
dataSet,labels=createDataSet()
entropy=calc_entropy(dataSet)
def splitDataSet(dataSet, axis, value):
    retdataSet=[]
    for data in dataSet:
        if data[axis]==value:
            subFeatures=data[:axis]
            subFeatures.extend(data[axis+1:])
            retdataSet.append(subFeatures)
    return retdataSet
            
def chooseBestFeatureToSplit(dataSet):
    feature_nums=len(dataSet[0])-1
    baseEntropy=calc_entropy(dataSet)
    best_infor_gain=0.0
    best_feature=-1
    for i in range(feature_nums):
        feature_list=[example[i] for example in dataSet]
        unique_value=set(feature_list)
        new_entropy=0.0
        for value in unique_value:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/len(dataSet)
            new_entropy+=prob*calc_entropy(subDataSet)
        infor_gain=baseEntropy-new_entropy
        if infor_gain>best_infor_gain:
            best_infor_gain=infor_gain
            best_feature=i
    return best_feature
best_feature= chooseBestFeatureToSplit(dataSet)           

def majorityCnt(classList):
    m=len(classList)
    class_nums={}
    for i in range(m):
        label=classList[i]
        class_nums[label]=class_nums.get(label,0)+1
    sorted_class_nums=sorted(class_nums.items(),key=lambda x:x[1],reverse=True)
    return sorted_class_nums[0][0]

#==ID3 algorithm===================
def createTree(dataSet,labels):
    label_list=[example[-1] for example in dataSet]
    if label_list.count(label_list[0])==len(label_list):
        return label_list[0]
    if len(dataSet[0])==1:
        return majorityCnt(label_list)
    best_feature=chooseBestFeatureToSplit(dataSet)
    best_label=labels[best_feature]
    my_tree={best_label:{}}
    feature_value=[example[best_feature] for example in dataSet]
    unique_value=set(feature_value)
    for value in unique_value:
        sublabels=labels[0:best_feature]
        sublabels.extend(labels[best_feature+1:])
        my_tree[best_label][value]=createTree(splitDataSet(dataSet, \
               best_feature, value),sublabels)
    return my_tree
my_tree=createTree(dataSet,labels) 

#========分类================ 
def classify(inputTree,feature_labels,testVec):
    first_str=list(inputTree.keys())[0]
    secondDict=inputTree[first_str]
    featureIndex=feature_labels.index(first_str)
    for key in secondDict.keys():
        if testVec[featureIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classlabel=classify(secondDict[key],feature_labels,testVec)
            else:
                classlabel=secondDict[key]
    return classlabel

result=classify(my_tree,labels,['old','no','no','well'])
    
    




        
            
            
    


