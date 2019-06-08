# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:56:11 2019

@author: 29132
"""

class BitNode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
        
class BitTree:
    def __init__(self):
        self.root=None
        self.bitlist=[]
        
    def add_bit_item(self,item):
        if not isinstance(item,BitNode):
            item=BitNode(item)
        if self.root is None:
            self.root=item
            self.bitlist.append(item)
        else:
            rootNode=self.bitlist[0]
            while True:
                if item.value<rootNode.value:
                    if rootNode.left==None:
                        rootNode.left=item
                        self.bitlist.append(item)
                        break
                    else:
                        rootNode=rootNode.left
                elif item.value>rootNode.value:
                    if rootNode.right==None:
                        rootNode.right=item
                        self.bitlist.append(item)
                        break
                    else:
                        rootNode=rootNode.right
    
    def preOrder(self,root):
        if root== None:
            return
        print(root.data)
        self.preOrder(root.left)
        self.preOrder(root.right)
        
    def pre_order(self,root):
        if root==None:
            return
        print(root.data)
        node=root
        my_stack=[]
        while node or my_stack:
            print(node.data)
            my_stack.append(node)
            node=node.left
        node=my_stack.pop()
        node=node.right
            