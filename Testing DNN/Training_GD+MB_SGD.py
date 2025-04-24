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
#from line_profiler import profile



#%% Set random seeds

#set seed for random training points
np.random.seed(42)

#set seed for random bias and weights
rand.seed(42)

#%% Define functions


#Define map being used:
def func_map(x, y):
    return(x**2 + y**2)

#Define activation function 
def sigma(y):
    return np.tanh(y)

#Define derivative of activation function 
def SigmaPrime(y):
    return  1 - np.tanh(y)**2


def residuals(theta, X, d):
    a_1,a_2,a_3,z_1,z_2 = DNN_FP(theta, X.T)
    residual = a_3.reshape(X.shape[0],) - d
    return residual


#DNN architechture function
def DNN_FP(theta, X):
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
    

    a_0 = X 
    #hidden layer 1
    z_1 = W_1 @ a_0
    a_1 = sigma(z_1.T + B_1).T
    #hidden layer 2
    z_2 = W_2 @ a_1
    a_2 = sigma(z_2.T + B_2).T
    #output layer
    a_3 = ((W_3 @ a_2).T + B_3).T
    
        
    return a_1, a_2, a_3, z_1, z_2

#@profile
#Backward Pass algorithm
def BP(theta, X, d):
    
    a_0 = X 
    a_1, a_2, a_3, z_1, z_2 = DNN_FP(theta, X)
    #Unpack inputs in theta 
    index = Nodes_1
    index += Nodes_2
    index += Nodes_3
    index += Nodes_1*2
    W_2 = theta[index:index+(Nodes_2*Nodes_1)].reshape(Nodes_2,Nodes_1)
    index += (Nodes_2*Nodes_1)
    W_3 = theta[index:index + (Nodes_3*Nodes_2)].reshape(Nodes_3,Nodes_2)
    
    
    #l = 3 
    delta_3 = -(d - a_3) 
    #Calculate partial derivatives  
    DJ_DW_3 = np.outer(delta_3,a_2)
            
    #l = 2
    sigma_prime_z_2 = SigmaPrime(z_2)
    W_3_delta_3 = W_3.T @ delta_3
    delta_2 = sigma_prime_z_2 * W_3_delta_3
        
    #Calculate partial derivatives
    DJ_DW_2 = np.outer(delta_2,a_1)
            
    #l = 1
    sigma_prime_z_1 = SigmaPrime(z_1)
    W_2_delta_2 = W_2.T @ delta_2
    delta_1 = sigma_prime_z_1 * W_2_delta_2
            
    #Calculate partial derivatives
    DJ_DW_1 = np.outer(delta_1,a_0)
    
    
    DJ_DW_1 = DJ_DW_1.flatten() #shape (2 x node 1)
    DJ_DW_2 = DJ_DW_2.flatten() #shape (Node 2 x node 1)
    DJ_DW_3 = DJ_DW_3.flatten() #shape (Node 3 x node 2)

    DJ_DW = np.concatenate([DJ_DW_1, DJ_DW_2, DJ_DW_3])
    DJ_DB = np.concatenate([delta_1, delta_2, delta_3])

    DJ = np.concatenate([DJ_DB, DJ_DW]) 
   
    
    return DJ

#@profile
#SGD algorthim
def MB_SGD(X, d, X_val, d_val, theta_initial, num_epochs, NB = 70, eta = 0.01):
    num_batches = int(X.shape[0]/NB)
    theta = np.zeros((num_epochs+1,num_batches+1,len(theta_initial)))
    theta[0][0] = theta_initial
    mse = np.zeros(num_epochs)
    mse_val = np.zeros(num_epochs)
    for j in range(1,num_epochs+1):
        #create random permutation for MB
        N = np.arange(X.shape[0])
        N_permutation = np.random.permutation(N)
        for k in range(1,num_batches+1):
            #compute gradient
            sum_mb = np.zeros(shape = theta_initial.shape)
            for m in range((k-1)*NB,k*NB):
                sum_mb += BP(theta[j-1][k-1], X[N_permutation[m-1]], d[N_permutation[m-1]].reshape(1,)) #do they need -1 s?
            p_k = 1/NB * sum_mb
            theta[j-1][k] = theta[j-1][k-1] - eta*p_k

        theta[j][0] = theta[j-1][num_batches]
      
        
        V = residuals(theta[j][0], X, d)
        mse[j-1] = 0.5 * np.mean( V ** 2)
        V_val = residuals(theta[j][0], X_val, d_val)
        mse_val[j-1] = 0.5 * np.mean( V_val ** 2)
        
        
    return theta[num_epochs][0] , mse, mse_val

#@profile
#GD optimisation algorithm
def GD(X, d, X_val, d_val, theta_initial, num_epochs, eta = 0.01):
    theta = np.zeros((num_epochs+1,len(theta_initial)))
    theta[0] = theta_initial
    mse = np.zeros(num_epochs)
    mse_val = np.zeros(num_epochs)
    for j in range(1,num_epochs+1):
        sum_grad = 0
        for i in range(X.shape[0]):
            sum_grad+= BP(theta[j-1], X[i], d[i].reshape(1,))
        p_j = sum_grad / X.shape[0]
        theta[j] = theta[j-1] - eta * p_j
        V = residuals(theta[j], X, d)
        mse[j-1] = 0.5 * np.mean( V ** 2)
        V_val = residuals(theta[j], X_val, d_val)
        mse_val[j-1] = 0.5 * np.mean( V_val ** 2)
        
        
    return theta[num_epochs], mse, mse_val
    
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
rand.seed()
np.random.seed()
        
#ask user for node sizes or set inputs
Nodes_1 = 10 # int(input("Input number of nodes for layer 1: "))   
Nodes_2 = 20 # int(input("Input number of nodes for layer 2: "))  
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
x1 = x[:,0]
x2 = x[:,1]
d = training_output



#%%Run the DNN - testing
#X = np.stack((x1,x2)).T
#A = np.stack((x1,x2))  

#V = residuals(initial_theta, x1, x2, d)

#a_1,a_2,a_3,z_1,z_2,z_3 = DNN_FP(initial_theta,x1,x2)
#print("solution = ", V)

#%%Run BP check 

X = np.array([[1,1]])
d[0] = 2

#X = np.stack((x1,x2)).T
DJ = BP(initial_theta, X[0], d[0].reshape(1,))

print(DJ)


#%% GD optimisation

x = training_points
d = training_output

x1 = x[:,0]
x2 = x[:,1]

num_epochs = 100
X = np.stack((x1,x2)).T

theta_optimised, GD_mse, GD_mse_val = GD(X, d, validation_points, validation_output, initial_theta, num_epochs)


plt.plot(GD_mse)

plt.plot(GD_mse_val)



#%% MB-SGD

## Use MB_SGD to optimise

x = training_points
d = training_output

x1 = x[:,0]
x2 = x[:,1]

num_epochs = 100
X = np.stack((x1,x2)).T

theta_optimised, MB_mse, MB_mse_val = MB_SGD(X, d, validation_points, validation_output, initial_theta, num_epochs)


plt.plot(MB_mse)

plt.plot(MB_mse_val)



#%%  plot DNN evaluated with optimised theta.

a_1,a_2,a_3,z_1,z_2 = DNN_FP(theta_optimised, X.T)


optimised_output_layer = a_3



#%% plot training data

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, optimised_output_layer, c=optimised_output_layer)




