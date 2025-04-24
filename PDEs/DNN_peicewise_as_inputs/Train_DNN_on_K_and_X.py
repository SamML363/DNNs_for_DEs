#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:39:55 2025

@author: samuellewis
"""


import torch
import scipy.io as sio

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import torch.nn as nn
import torch.optim as optim
import os
#from sklearn.model_selection import train_test_split


#%% functions for reading and writing from/to files


def read_txt(txt_file):
    f = open(txt_file, 'r')
    content = f.read()
    lines = content.splitlines()
    dat = [x.split(',') for x in lines]
    dat = np.array(dat,dtype=float)
    f.close()
    return dat[0] if len(dat) == 1 else dat


def write_txt(data, filename):
    with open(filename, 'w') as f:
        # Ensure `data` is 2D for iteration
        if data.ndim == 1:  # Handle 1D arrays
            data = data[:, np.newaxis]  # Add a new axis to make it 2D

        for row in data:
            row_string = ','.join(map(str, row))  # Convert row elements to strings
            f.write(row_string + '\n')  # Write row to file
            

#%% Prepare data as tensors


X, Y = read_txt(r"X.txt"), read_txt(r"Y.txt")
X_val, Y_val = read_txt(r"X_val.txt"), read_txt(r"Y_val.txt")
coords = read_txt(r"coords.txt")


#check shape of arrays
print(X.shape)
print(X_val.shape)
print(Y.shape)
print(Y_val.shape)


#set seeds
np.random.seed(2)
torch.manual_seed(2)



#Preform normalisation between 0 and 1
X_min = X.min()
X_max = X.max()
Y_min = Y.min()
Y_max = Y.max()

X_array =  (X-X_min)/(X_max-X_min)
X_val_array = (X_val-X_min) / (X_max-X_min)
Y_array =  (Y-Y_min)/(Y_max-Y_min)
Y_val_array = (Y_val-Y_min)/(Y_max-Y_min)


#%% Experemental step of changing inputs X to have k values and x,y coordinates

#initialise number of mesh grid points and number of points for training data set.
training_size = X_array.shape[0]
mesh_size = Y_array.shape[1]

#restructure inputs and outputs for Individual sample based training with new size x = meshgrid * training data sizes
coords_tiled = np.tile(coords, (training_size, 1))  # Shape: (x, 2)
k_vals_repeated = np.repeat(X_array, mesh_size, axis=0)  # Shape: (x, 9)
u_values = Y_array.flatten().reshape(-1, 1)  # Shape: (x, 1)

# Step 4: Concatenate inputs
X = np.hstack((k_vals_repeated, coords_tiled))  # Shape: (x 11)
Y = u_values  # Shape: (x, 1)




#%%
#preform the same change for validation data set

validation_size = X_val_array.shape[0]
coords_tiled = np.tile(coords, (validation_size, 1))
k_vals_repeated = np.repeat(X_val_array, mesh_size, axis=0)  
u_values = Y_val_array.flatten().reshape(-1, 1)  
X_val = np.hstack((k_vals_repeated, coords_tiled))  
Y_val = u_values  

#%%

# Convert to 2D PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)

print(X.shape)

#%% Create the surrogate model
 
"""
# Define the DNN
model = nn.Sequential(
    nn.Linear(X.shape[1], 64),
    nn.Tanh(),
    nn.Linear(64, 256),
    nn.Tanh(),
    nn.Linear(256, 1024),
    nn.Tanh(),
    nn.Linear(1024, 2048),
    nn.Tanh(),
    nn.Linear(2048, Y.shape[1])
)
"""

# Define the DNN
model = nn.Sequential(
    nn.Linear(X.shape[1], 10),
    nn.Tanh(),
    nn.Linear(10, 15),
    nn.Tanh(),
    nn.Linear(15, Y.shape[1])
)


"""
# Define the DNN
model = nn.Sequential(
    nn.Linear(X.shape[1], 10),
    nn.Sigmoid(),
    nn.Linear(10, 20),
    nn.Sigmoid(),
    nn.Linear(20, 15),
    nn.Sigmoid(),
    nn.Linear(15, 30),
    nn.Sigmoid(),
    nn.Linear(30, Y.shape[1])
)
"""
 
# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0002)
#optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
 
n_epochs = 3000  # number of epochs to run
plot_epoch = 100
batch_size =  int(X.shape[0] /10) # size of each batch
print("batch_size", batch_size)
print("batches = ", X.shape[0] / batch_size)


#Save for plotting
mse_vec= np.zeros(int(n_epochs/plot_epoch))
mse_val_vec= np.zeros(int(n_epochs/plot_epoch))
x_epoch= np.zeros(int(n_epochs/plot_epoch))
index = 0

##Outer Loop Epochs
for epoch in range(n_epochs):
    
    permutation = torch.randperm(X.size()[0])       
    model.train()
    #Inner loop for bacthes
    for i in range(0,X.size()[0], batch_size):   
        indices = permutation[i:i+batch_size]
        X_batch, Y_batch = X[indices], Y[indices]
        
        # forward pass
        Y_pred = model(X_batch)
        
        #testing
        if torch.any(torch.isnan(Y_pred)):
            print(f"NaN values detected in Y_pred at epoch {epoch}")
        
        loss = loss_fn(Y_pred, Y_batch)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # update parameter
        optimizer.step()
            # print progress
            
    # evaluate accuracy at end of each epoch
    if epoch % plot_epoch == 0:
        print(epoch)
        model.eval()
        with torch.no_grad():
            Y_pred = model(X)

        with torch.no_grad():
            Y_pred_val = model(X_val)

        mse = loss_fn(Y_pred, Y)
        mse_val = loss_fn(Y_pred_val, Y_val)
        
        #print(f"Epoch {epoch}, MSE: {loss.item()}")

        mse_vec[index] = mse.item()
        mse_val_vec[index] = mse_val.item()
        x_epoch[index] = epoch
        index+= 1






#%%plot mse vals


plt.figure(figsize=(8, 6))
plt.semilogy(x_epoch,mse_vec,label="Training")
plt.semilogy(x_epoch,mse_val_vec,label="Validation")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Epochs")
plt.title("Learning rate = 0.0002" )
plt.show()

#plt.savefig("mse.png")




#%% plot true solution for reference
i = 3

x = coords[:,0]
y = coords[:,1]
z = Y_val_array[i]
z = z*(Y_max - Y_min) + Y_min
#z = (( np.array(z.detach().numpy()) ))

print(X_val_array[i]*(X_max - X_min) + X_min)

# 3d plot
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(x, y, z, c=z)
contour = plt.tricontourf(x, y, z, levels=20, cmap=cm.viridis)
plt.colorbar(contour)
plt.show()

#%%
# Success for comparisson with X_val

x = coords[:,0]
y = coords[:,1]
test_k = X_val_array[i]  #*(X_max - X_min) + X_min
test_k = test_k[:, np.newaxis] 
test_k_tiled = np.tile(test_k, (1, coords.shape[0]))

test_input = np.vstack((test_k_tiled, coords.T)).T
test_input = torch.tensor(test_input, dtype=torch.float32)

z = np.zeros(mesh_size)
for index in range(mesh_size):
    z[index] = model(test_input[index])
    #print(z[index])

z = z*(Y_max - Y_min) + Y_min
#z = (( np.array(z.detach().numpy()) ))
#print(X_val[i]*(X_max - X_min) + X_min)


# 3d plot
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(x, y, z, c=z)
contour = plt.tricontourf(x, y, z, levels=20, cmap=cm.viridis)
plt.colorbar(contour)
plt.show()



