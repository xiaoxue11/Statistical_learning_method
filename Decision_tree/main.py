# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:45:07 2019

@author: 29132
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
print(cross_val_score(clf, iris.data, iris.target, cv=10))
clf.fit(iris.data,iris.target)
y_pred=clf.predict(iris.data)
accuracy=clf.score(iris.data,iris.target)