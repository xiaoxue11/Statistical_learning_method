# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:49:51 2019

@author: 29132
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

class perceptron_model:
    def __init__(self,learning_rate):
        self.lr=learning_rate
    def fit(self,X,y):
        m,n=X.shape
        self.W=np.zeros(n)
        self.b=0
        is_wrong = True
        while is_wrong:
            wrong_count = 0
            for i in range(m):
                X1 = X[i]
                y1 = y[i]
                if np.sum((y1 * (np.dot(X1, self.W)+self.b))) <= 0:
                    self.W = self.W + self.lr*np.dot(X1, y1)
                    self.b = self.b + self.lr*y1
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = False
        return self.W,self.b
    
iris=load_iris()
X=iris['data']
y=iris['target']
X_train=X[:100,[0,1]]
y_train=y[:100]
y=np.array([1 if i ==1 else -1 for i in y_train])

clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
clf.fit(X_train,y_train)
print(clf.coef_)
x2_ponits = np.arange(4, 8).reshape(-1,1)
y2_ = -(clf.coef_[0][0]*x2_ponits + clf.intercept_)/clf.coef_[0][1]

model=perceptron_model(learning_rate=0.01)
W,b=model.fit(X_train,y)
x3_ponits = np.arange(4, 8).reshape(-1,1)
y3_ = -(W[0]*x3_ponits + b)/W[1]


plt.plot(X[0:50,0],X[0:50,1],'ro')
plt.plot(X[50:100,0],X[50:100,1],'yx')
plt.plot(x2_ponits, y2_)
plt.plot(x3_ponits, y3_)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(['0','1','sklearn_perceptron','defunction_by_user'],loc='upper left')









        