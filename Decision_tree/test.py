# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:37:29 2019

@author: 29132
"""

a=[1,2,3]
b=[4,5,6]
a.append(b)
print(a)
a.extend(b)
print(a)

#%%
import matplotlib.pyplot as plt
## equivalent but more general
ax1=plt.subplot(221)
## add a subplot with no frame
ax2=plt.subplot(222, frameon=False)
## add a polar subplot
plt.subplot(223, projection='polar')
## add a red subplot that shares the x-axis with ax1
plt.subplot(224, sharex=ax1, facecolor='red')
##delete ax2 from the figure
plt.delaxes(ax2)
##add ax2 to the figure again
plt.subplot(ax2)

#%%====
import numpy as np
x = np.arange(1000)
y = np.sin(x)

for i in range(5):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    plt.close(fig)
print(plt.get_fignums())

#%%=========================
d = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four',
     5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'}
k=list(d.keys())[0]
print(k)

#%%====
s={1:'no',2:{0:'no',1:'yes'}}
for key in s.keys():
    print(type(s[key]).__name__)
    
#%%
class BitNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right=None

class BitTree:
    def __init__(self):
        self.root=None
        self.bitlist=[]
        
    def add_bit_item(self,item):
        if not isinstance(item,BitNode):
            item=BitNode(item)
        if self.root==None:
            self.root=item
            self.bitlist.append(item)
        else:
            rootNode=self.bitlist[0]
            while True:
                if item.data<rootNode.data:
                    if rootNode.left==None:
                        rootNode.left=item
                        self.bitlist.append(item)
                        break
                    else:
                        rootNode=rootNode.left
                elif item.data>rootNode.data:
                    if rootNode.right==None:
                        rootNode.right=item
                        self.bitlist.append(item)
                        break
                    else:
                        rootNode=rootNode.right
node1=BitNode(15)
node2=BitNode(9)
node3=BitNode(8)
node4=BitNode(16)
bit_tree=BitTree()
for node in [node1,node2,node3,node4]:
    bit_tree.add_bit_item(node)

print(bit_tree.root.left.data)
    

