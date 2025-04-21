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
from line_profiler import profile


#Goal - check optimised theta - produces good scatter graph  done




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
    return np.tanh(y)


def residuals(theta, x1, x2, d):
    residual = DNN(theta, x1, x2) - d
    return residual


#DNN architechture function
@profile
def DNN(theta, x1, x2):
    # a_0, W_1, W_2, W_3 , B_1, B_2, B_3
    
    #Unpack inputs in theta 
    B_1 = theta[:Nodes_1]
    index = Nodes_1
    B_2 = theta[index: index + Nodes_2]
    index += Nodes_2
    B_3 = theta[index:index+Nodes_3]
    index += Nodes_3
    W_1 = theta[index:index+(Nodes_1*2)].reshape(Nodes_1,2)
    index += Nodes_1*2
    W_2 = theta[index:index+(Nodes_2*Nodes_1)].reshape(Nodes_2,Nodes_1)
    index += (Nodes_2*Nodes_1)
    W_3 = theta[index:index + (Nodes_3*Nodes_2)].reshape(Nodes_3,Nodes_2)
    

    a_0 = np.stack((x1,x2))

    #hidden layer 1
    z_1 = W_1 @ a_0
    a_1 = sigma(z_1.T + B_1).T
    #hidden layer 2
    z_2 = W_2 @ a_1
    a_2 = sigma(z_2.T + B_2).T
    #output layer
    z_3 = W_3 @ a_2
    a_3 = (z_3.T + B_3).T
    
    output = a_3.reshape(len(x1),)

        
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
Nodes_1 = 6 # int(input("Input number of nodes for layer 1: "))
Nodes_2 = 7 # int(input("Input number of nodes for layer 2: "))  
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
#V = residuals(initial_theta, x1, x2, d)

V = DNN(initial_theta,x1,x2)
#print("solution = ", V)




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