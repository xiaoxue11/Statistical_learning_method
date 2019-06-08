# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:46:09 2019

@author: 29132
"""

import numpy as np
a=np.array([1,2,3])
rs1=np.tile(a,2)
rs2=np.tile(a,(1,2))
rs3=np.tile(a,(2,1))

b=np.array([[1,2],[3,4]])
rs11=np.tile(b,2)
rs22=np.tile(b,(1,2))
rs33=np.tile(b,(2,1))

#argsort
#1 dimension
x=np.array([3,1,0,2])
inx=np.argsort(x)
print(x[inx])

#===2-dimension
y=np.array([[1,0],[4,2]])
ind=np.argsort(y)
ind0=np.argsort(y,axis=0)
ind1=np.argsort(y,axis=1)
#%%
from knn import createDataset,knn_classfier
import numpy as np
X=np.array([0,0])
group,labels=createDataset()
res=knn_classfier([0,0], group, labels, 3)