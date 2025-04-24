#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 18:06:51 2025

@author: samuellewis
"""

#import packages
import numpy as np
import matplotlib.pyplot as plt
import os



#Define map being used:
def func_map(x, y):
    return np.array([(np.sin(np.pi * x_i) * np.sin(2 * np.pi * y_i)) if 
            (x_i < 0 or y_i < 0) else 1 for x_i, y_i in zip(x, y)])


#%% create data

#set seed for random training points
np.random.seed(42)

#create meshgrid of points
x = np.linspace(-1,1,200)
y = np.linspace(-1,1,200)
xx, yy = np.meshgrid(x,y)


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
num_training_points = 30000
indices = np.random.choice(len(mesh_points), num_training_points, replace=False)
training_points = mesh_points[indices]
training_output = func_map(training_points[:,0], training_points[:,1])


#Create list of validation points

validation_indices = np.setdiff1d(np.arange(len(mesh_points)), indices)
validation_points = mesh_points[validation_indices]
validation_output = func_map(validation_points[:,0], validation_points[:,1])


#%% write data to file



# Separate features (X) and targets (Y) for training
X_train = training_points #np.array([point[0] for point in training_points])  # Features
Y_train = training_output #np.array([point[1] for point in training_points])  # Targets

# Separate features (X_val) and targets (Y_val) for validation
X_val = validation_points #np.array([point[0] for point in validation_points])  # Features
Y_val = validation_output #np.array([point[1] for point in validation_points])  # Targets


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
