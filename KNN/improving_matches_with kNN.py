# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:05:59 2019

@author: 29132
"""
import numpy as np
from knn import file2matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
from knn import knn_classfier
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (7.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
#%%====load data and transform y 
X,y=file2matrix('datingTestSet.txt')
le = preprocessing.LabelEncoder()
y_value=le.fit_transform(y)
#%% 
plt.figure()
plt.scatter(X[:,0],X[:,1],c=np.squeeze(y_value),cmap=plt.cm.Spectral)
plt.xlabel('Number of frequent flyer miles earned per year')
plt.ylabel('Percentage of time spent playing video games')
plt.figure()
plt.scatter(X[:,0],X[:,2],c=np.squeeze(y_value),cmap=plt.cm.Spectral)
plt.figure()
plt.scatter(X[:,1],X[:,2],c=np.squeeze(y_value),cmap=plt.cm.Spectral)
#%%==============
X_scale=preprocessing.scale(X)
X_train,X_test,y_train,y_test=train_test_split(X_scale,y,test_size=0.1,random_state=42)
m=X_test.shape[0]
right_count=0
for i in range(m):
    y_pred=knn_classfier(X_test[i],X_train,y_train,3)
    if y_pred==y_test[i]:
        right_count+=1
accuracy=right_count/len(y_test)

#%%sklearn method to deal with this problem
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

    
