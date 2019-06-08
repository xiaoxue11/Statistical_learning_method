# -*- coding: utf-8 -*-
"""
Created on Thu May 30 23:42:24 2019

@author: 29132
"""

A=[1,2,3,4]
B=[1,1,1,1]
C=[1]*len(A)
D=[i-j for i,j in zip(C,A)]
print(sum([i*j for i,j in zip(A,B)]))
print(sum([i*j for i,j in zip(D,B)]))