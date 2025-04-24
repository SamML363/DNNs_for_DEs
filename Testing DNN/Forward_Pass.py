#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:33:51 2024

@author: samuellewis
"""

import numpy as np
import random as rand


#%%


#set seed for random bias and weights
rand.seed(42)

#Sigma(y) = ReLU(y)
def sigma(y):
    return max(0,y)
    

#DNN architechture function
def DNN(a_0, W_1, W_2, W_3 , B_1, B_2, B_3, Nodes_1, Nodes_2 ):
    
    a_1 = ([])
    a_2 = ([])
    a_3 = ([])
    
    for j in range(Nodes_1):
        
        z_l = 0
        for i in range(2):
        
            z_l += W_1[j][i]*a_0[i]
        
        a_1.append(sigma(z_l + B_1[j]))
        
        print(a_1)
       
    for j in range(Nodes_2):
        
        z_l = 0
        for i in range(Nodes_1):
        
            z_l += W_2[j][i]*a_1[i]
        
        
        a_2.append(sigma(z_l+B_2[j])) 
        
        print(a_2)
        
    for j in range(2):
        
        z_l = 0
        for i in range(Nodes_2):
            
         
            z_l += W_3[j][i] * a_2[i]
        
            
        
        
        a_3.append(z_l + B_3[j]) 
        
        print(a_3)
    
    return a_3
        
   
#ask user for node sizes
Nodes_1 = int(input("Input number of nodes for layer 1: "))
Nodes_2 = int(input("Input number of nodes for layer 2: "))  
  

#Initialize weights sizes
W_1 = np.zeros((Nodes_1 ,2))
W_2 = np.zeros((Nodes_2,Nodes_1))
W_3 = np.zeros((2,Nodes_2))

for i in range(W_1.shape[0]):
    for j in range (W_1.shape[1]):
        W_1[i][j] = rand.random() * 10

for i in range(W_2.shape[0]):
    for j in range (W_2.shape[1]):
        W_2[i][j] = rand.random() * 10
        
for i in range(W_3.shape[0]):
    for j in range (W_3.shape[1]):
        W_3[i][j] = rand.random() * 10


#Initialize bias
B_1 = np.zeros(Nodes_1)
B_2 = np.zeros(Nodes_2)
B_3 = np.zeros(2)

for i in range(len(B_1)):
    B_1[i] = rand.random() * 10
    
for i in range(len(B_2)):
    B_2[i] = rand.random() * 10

for i in range(len(B_3)):
    B_3[i] = rand.random() * 10


a_0 = np.zeros(2)
a_0[0] = 1
a_0[1] = 2



#Run the DNN
V = DNN(a_0, W_1, W_2, W_3 , B_1, B_2, B_3, Nodes_1, Nodes_2 )

print("solution = ", V)

