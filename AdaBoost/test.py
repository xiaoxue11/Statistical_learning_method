# -*- coding: utf-8 -*-
"""
Created on Tue May 28 23:46:52 2019

@author: 29132
"""

import numpy as np
retarray=np.ones([2,1])
data=np.array([[1,2,3,4,5],[-1,1,2,3,1]])
m=data[:,0]<1
retarray[m]=-1.0