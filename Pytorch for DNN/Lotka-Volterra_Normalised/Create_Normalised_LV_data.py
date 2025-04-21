#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:06:51 2025

@author: samuellewis
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp



#Define map being used:
def LV_equations(time, XY, gamma, delta, alpha = 1.0, beta = 0.4 ):
    x,y = XY
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    
    return dx_dt, dy_dt


#%% create data

#set seed for random training points
np.random.seed(42)

#create meshgrid of points
gamma = np.linspace(0.5,2,200)
delta = np.linspace(0.5,1,200)
gg, dd = np.meshgrid(gamma,delta)


#Initial conditions
x_0 = 6
y_0 = 9
t_interval = (0,15)

#Initialize vectors to store resuts
final_x = np.zeros(len(gamma)*len(delta))
final_y = np.zeros(len(gamma)*len(delta))
index = 0


for g in gamma:
    for d in delta:

        #Solve LV equations
        solution = solve_ivp(LV_equations, t_interval, [x_0,y_0], args=(g,d))
            

        #extract time data and population data
        x,y = solution.y
        
        #store values at t = T
        final_x[index] = x[-1]
        final_y[index] = y[-1]
        index+= 1
            
        
#flatten arrays 
flat_gg = gg.flatten()
flat_dd = dd.flatten()

#%%Create plot for QoI_X

#Create scatter plot
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(flat_gg, flat_dd, final_x, c=final_x)
ax1.set_xlabel(r"$\gamma$", fontsize = 14)
ax1.set_ylabel(r"$\delta$", fontsize = 14)
ax1.set_zlabel("population x", fontsize = 14)
ax1.set_title("Training data for population x", fontsize=16)
plt.savefig("LV_QoI_x_training.pdf", format="pdf", bbox_inches="tight")


#%%Create plot for QoI_y

#Create scatter plot
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(flat_gg, flat_dd, final_y, c=final_y)
ax1.set_xlabel(r"$\gamma$", fontsize = 14)
ax1.set_ylabel(r"$\delta$", fontsize = 14)
ax1.set_zlabel("population y", fontsize = 14)
ax1.set_title("Training data for population y", fontsize=16)
plt.savefig("LV_QoI_y_training.pdf", format="pdf", bbox_inches="tight")

#%% shape data to be written to file


#Combine flattend arrays
mesh_points = np.column_stack((flat_gg, flat_dd))

#Choose points for training
num_training_points = 30000
indices = np.random.choice(len(mesh_points), num_training_points, replace=False)
training_points = mesh_points[indices]
training_output = np.stack((final_x[indices], final_y[indices]), axis = 1)


#Create list of validation points

validation_indices = np.setdiff1d(np.arange(len(mesh_points)), indices)
validation_points = mesh_points[validation_indices]
validation_output = np.stack((final_x[validation_indices], final_y[validation_indices]), axis = 1)


#%% write data to file


# Separate features (X) and targets (Y) for training
X_train = training_points #np.array([point[0] for point in training_points])  # Features
Y_train = training_output #np.array([point[1] for point in training_points])  # Targets

# Separate features (X_val) and targets (Y_val) for validation
X_val = validation_points #np.array([point[0] for point in validation_points])  # Features
Y_val = validation_output #np.array([point[1] for point in validation_points])  # Targets

#%% normalis the data and then right to file:

#normalise--------------
L = -1
U = 1

#x1 input 1
x1_min = np.min(X_train[:,0])
x1_max = np.max(X_train[:,0])
X_train[:,0] = L + (U-L)*(X_train[:,0]-x1_min)/(x1_max - x1_min)
x1_min_val = np.min(X_val[:,0])
x1_max_val = np.max(X_val[:,0])
X_val[:,0] = L + (U-L)*(X_val[:,0] -x1_min)/(x1_max - x1_min)
#x2 input 2
x2_min = np.min(X_train[:,1])
x2_max = np.max(X_train[:,1])
X_train[:,1] = L + (U-L)*(X_train[:,1] -x2_min)/(x2_max - x2_min)
x2_min_val = np.min(X_val[:,1])
x2_max_val = np.max(X_val[:,1])
X_val[:,1] = L + (U-L)*(X_val[:,1] -x2_min)/(x2_max - x2_min)
#y1 output 1
y1_min = np.min(Y_train[:,0])
y1_max = np.max(Y_train[:,0])
Y_train[:,0] = L + (U-L)*(Y_train[:,0] -y1_min)/(y1_max - y1_min)
y1_min_val = np.min(Y_val[:,0])
y1_max_val = np.max(Y_val[:,0])
Y_val[:,0] = L + (U-L)*(Y_val[:,0] -y1_min)/(y1_max - y1_min)
#y1 output 2
y2_min = np.min(Y_train[:,1])
y2_max = np.max(Y_train[:,1])
Y_train[:,1] = L + (U-L)*(Y_train[:,1] -y2_min)/(y2_max - y2_min)
y2_min_val = np.min(Y_val[:,1])
y2_max_val = np.max(Y_val[:,1])
Y_val[:,1] = L + (U-L)*(Y_val[:,1] -y2_min)/(y2_max - y2_min)

def write_txt(data, filename):
    with open(filename, 'w') as f:
        # Ensure `data` is 2D for iteration
        if data.ndim == 1:  # Handle 1D arrays
            data = data[:, np.newaxis]  # Add a new axis to make it 2D

        for row in data:
            row_string = ','.join(map(str, row))  # Convert row elements to strings
            f.write(row_string + '\n')  # Write row to file
            
# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.realpath(__file__))

# File paths (in the same directory as the script)
x_file = os.path.join(script_directory, "X.txt")
y_file = os.path.join(script_directory, "Y.txt")
x_val_file = os.path.join(script_directory, "X_val.txt")
y_val_file = os.path.join(script_directory, "Y_val.txt")

# Write the arrays to text files
write_txt(X_train, 'X.txt')    # Training features
write_txt(Y_train, 'Y.txt')    # Training targets
write_txt(X_val, 'X_val.txt')  # Validation features
write_txt(Y_val, 'Y_val.txt')  # Validation targets
