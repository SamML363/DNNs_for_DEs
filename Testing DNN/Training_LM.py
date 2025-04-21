#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:33:51 2024

@author: samuellewis
"""
#Import required libraries
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


#Goal - check optimised theta - produces good scatter graph  done

#unsuccessfull  - mse plot

# need to test against validation data

"""
plt.savefig('ReLU Activation.png', dpi=300)  
plt.close()
"""


#%% Set random seeds

#set seed for random training points
np.random.seed(42)

#set seed for random bias and weights
rand.seed(42)

#%% Define functions


#Define map being used:
def func_map(x, y):
    return(x**2 + y**2)

#Define activation function (ReLU or tanh?)
def sigma(y):
    return max(0,y) #np.tanh(y)

def residuals(theta, x1, x2, d):
    residual = DNN(theta, x1, x2) - d
    return residual


#DNN architechture function
def DNN(theta, x1, x2):
    # a_0, W_1, W_2, W_3 , B_1, B_2, B_3
    #Nodes_1 = 2
    #Nodes_2 = 3
    #Nodes_3 = 1
    
    #Unpack inputs in theta - not especially efficient
    
    B_1 = np.zeros(Nodes_1) 
    B_2 = np.zeros(Nodes_2) 
    B_3 = np.zeros(Nodes_3)  
    
    W_1 = np.zeros((Nodes_1 ,2))
    W_2 = np.zeros((Nodes_2,Nodes_1))
    W_3 = np.zeros((Nodes_3,Nodes_2))
    
    for i in range(len(B_1)):
        B_1[i] = theta[i]
    for i in range(len(B_2)):
        B_2[i] = theta[i+Nodes_1]
    for i in range(len(B_3)):
        B_3[i] = theta[i+Nodes_1+Nodes_2]   
        
    no_cols = W_1.shape[1]
    for i in range(W_1.shape[0]*W_1.shape[1]):
        W_1[i//no_cols][i%no_cols] = theta[i+Nodes_1+Nodes_2+Nodes_3]   
    no_cols = W_2.shape[1]
    for i in range(W_2.shape[0]*W_2.shape[1]):
        W_2[i//no_cols][i%no_cols] =  theta[i+Nodes_1+Nodes_2+Nodes_3+(2*Nodes_1)]  
    no_cols = W_3.shape[1]
    for i in range(W_3.shape[0]*W_3.shape[1]):
        W_3[i//no_cols][i%no_cols] = theta[i+Nodes_1+Nodes_2+Nodes_3+(2*Nodes_1)+(Nodes_2*Nodes_1)] 
            
        
    
    """
    B_1 = ([])
    B_2 = ([])
    B_3 = ([])
    for i in range (2, len(theta)):
        if i < 2 + Nodes_1:
            B_1.append(theta[i])
        elif i < 2 + Nodes_1 + Nodes_2:
            B_2.append(theta[i])
        elif i < 2 + Nodes_1 + Nodes_2 + 2:
            B_3.append(theta[i])
    """
    
    output = np.zeros(len(x1))
    for index in range(len(x1)):
        
        
        a_1 = ([])
        a_2 = ([])
        a_3 = ([])
        
        a_0 = np.zeros(2) # only works if input is 2-D
        a_0[0] = x1[index]
        a_0[1] = x2[index]
        
        for j in range(Nodes_1):
            
            z_l = 0
            for i in range(2):
            
                z_l += W_1[j][i]*a_0[i]
            
            a_1.append(sigma(z_l + B_1[j]))
            
           
        for j in range(Nodes_2):
            
            z_l = 0
            for i in range(Nodes_1):
            
                z_l += W_2[j][i]*a_1[i]
            
            
            a_2.append(sigma(z_l+B_2[j])) 
            
            
        for j in range(Nodes_3):
            
            z_l = 0
            for i in range(Nodes_2):
                
             
                z_l += W_3[j][i] * a_2[i]
            
            
            a_3.append(z_l + B_3[j]) 
        
        
        output[index] = a_3[0] # only works if a_3 is 1 dimensional
    
    return output

#%% Create sample Data


#set seed for random training points
np.random.seed(42)

#create meshgrid of points
x = np.linspace(0,1,30)
y = np.linspace(0,1,30)
xx, yy = np.meshgrid(x,y)

#create contor plot
#z = func_map(xx, yy)
#h = plt.contourf(x,y,z)
#plt.show()


#flatten arrays 
flat_xx = xx.flatten()
flat_yy = yy.flatten()


#Create scatter plot
z = func_map(flat_xx, flat_yy)
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(flat_xx, flat_yy, z, c=z)


#Combine flattend arrays
mesh_points = np.column_stack((flat_xx, flat_yy))

#Choose points for training
num_training_points = 700
indices = np.random.choice(len(mesh_points), num_training_points, replace=False)
training_points = mesh_points[indices]
training_output = func_map(training_points[:,0], training_points[:,1])


#Create list of validation points

validation_indices = np.setdiff1d(np.arange(len(mesh_points)), indices)
validation_points = mesh_points[validation_indices]
validation_output = func_map(validation_points[:,0], validation_points[:,1])


#%% Create first guesses for Inputs Theta and evaluate DNN


#set seed for random bias and weights
rand.seed(42)
np.random.seed(42)
        
#ask user for node sizes or set inputs
Nodes_1 = 2 # int(input("Input number of nodes for layer 1: "))
Nodes_2 = 2 # int(input("Input number of nodes for layer 2: "))  
Nodes_3 = 1

#Initialize weights sizes
W_1 = np.zeros((Nodes_1 ,2))
W_2 = np.zeros((Nodes_2,Nodes_1))
W_3 = np.zeros((Nodes_3,Nodes_2))

for i in range(W_1.shape[0]):
    for j in range (W_1.shape[1]):
        W_1[i][j] = np.random.normal() 

for i in range(W_2.shape[0]):
    for j in range (W_2.shape[1]):
        W_2[i][j] = np.random.normal()
        
for i in range(W_3.shape[0]):
    for j in range (W_3.shape[1]):
        W_3[i][j] = np.random.normal()
        
W_1 = W_1.flatten() #2 x 2  (2 x node 1)
W_2 = W_2.flatten() # 3 x 2 (Node 2 x node 1)
W_3 = W_3.flatten() # 2 x 3 (2 x node 2)

W = np.concatenate([W_1, W_2, W_3])

#Initialize bias
B_1 = np.zeros(Nodes_1) #2
B_2 = np.zeros(Nodes_2) #3
B_3 = np.zeros(Nodes_3)       #2

B = np.concatenate([B_1, B_2, B_3])

""" Biases are Inialized as zero vectors
for i in range(len(B_1)):
    B_1[i] = rand.random() * 10
    
for i in range(len(B_2)):
    B_2[i] = rand.random() * 10

for i in range(len(B_3)):
    B_3[i] = rand.random() * 10
"""

a_0 = np.zeros(2) 
a_0[0] = 0.5
a_0[1] = 1

initial_theta = np.concatenate([B, W]) 


# test - single point
"""
x = a_0
d = training_output[0]

x1 = np.zeros(1)
x1[0] = 0.5
x2 = np.zeros(1)
x2[0] = 1.0
"""

# test vector
x = training_points
d = training_output
x1 = x[:,0]
x2 = x[:,1]

#Run the DNN
V = residuals(initial_theta, x1, x2, d)

#V = DNN(initial_theta,x1,x2)
print("solution = ", V)



#%% Optimise Theta using Training data

## Scipy.optimise.least_squares needs a function as input that returns vector 
#of residuals and an input theta that is the inital guess 
#I need to turn theta into a 1-d array 


x = training_points
d = training_output

x1 = x[:,0]
x2 = x[:,1]


theta_optimised = least_squares(residuals, initial_theta, method = 'lm', args=(x1, x2, d))



#%%  plot DNN evaluated with optimised theta.

optimised_DNN = DNN(theta_optimised.x, x1, x2)

# Create contor plot by switching data to meshgrids: difficult?


"""
X1, X2 = np.meshgrid(x1, x2)
Z = np.reshape(optimised_DNN, X1.shape)
"""

# - delete later is an example of scatter plot?? 
# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, optimised_DNN, c=optimised_DNN)



#%% plot mse values

#Run the DNN
V = residuals(theta_optimised.x, x1, x2, d)

mse = 0.5 * np.mean( V ** 2)

#print("solution = ", V)

print("MSE = ", mse)



#%% compare with validation data

x_val = validation_points
d_val = validation_output
x_val1 = x_val[:,0]
x_val2 = x_val[:,1]


validation_residuals = residuals(theta_optimised.x, x_val1, x_val2, d_val)

val_mse =  0.5 * np.mean( validation_residuals ** 2)

print(val_mse)

#%% plot validation data?

val_DNN = DNN(theta_optimised.x, x_val1, x_val2)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_val1, x_val2, val_DNN, c=val_DNN)

