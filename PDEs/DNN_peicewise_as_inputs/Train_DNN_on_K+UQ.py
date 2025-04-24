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
X_test, Y_test = read_txt(r"X_test.txt"), read_txt(r"Y_test.txt")
coords = read_txt(r"coords.txt")


#check shape of arrays
print(X.shape)
print(X_val.shape)
print(Y.shape)
print(Y_val.shape)
print(X_test.shape)
print(Y_test.shape)

#set seeds
np.random.seed(2)
torch.manual_seed(2)


# Convert to 2D PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)


#check for invalid entries
if  torch.any(torch.isinf(Y)):
    print(" infinite values detected in Y")



#Preform normalisation between 0 and 1
X_min = X.min()
X_max = X.max()
Y_min = Y.min()
Y_max = Y.max()

X =  (X-X_min)/(X_max-X_min)
X_val = (X_val-X_min) / (X_max-X_min)
Y =  (Y-Y_min)/(Y_max-Y_min)
Y_val = (Y_val-Y_min)/(Y_max-Y_min)

X_test = (X_test-X_min) / (X_max-X_min)




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
    nn.Tanh(),
    nn.Linear(10, 20),
    nn.Tanh(),
    nn.Linear(20, 15),
    nn.Tanh(),
    nn.Linear(15, 30),
    nn.Tanh(),
    nn.Linear(30, Y.shape[1])
)


"""
# Define the DNN
model = nn.Sequential(
    nn.Linear(X.shape[1], 64),
    nn.Sigmoid(),
    nn.Linear(64, 256),
    nn.Sigmoid(),
    nn.Linear(256, 1024),
    nn.Sigmoid(),
    nn.Linear(1024, 2048),
    nn.Sigmoid(),
    nn.Linear(2048, Y.shape[1])
)
"""
 
# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.003)
#optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
 
n_epochs = 3000   # number of epochs to run
plot_epoch = 100
batch_size = int(X.shape[0] /50) # size of each batch
print("batch_size ", batch_size)
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

#plt.savefig("k DNN learning curve simple network.pdf", format="pdf")

#%% test the time taken for evaluating the test (MC) data set

import time

start_time = time.time()

Y_MC_evaluations = model(X_test)


end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.6f} seconds")

#%% Plot hystograms of UQ for comoparisson

#pick index to compare
index = 480  # index used for UQ = 80,180,280,...
print(coords[index])

#undo normalisation of test outputs:
#Y_test = Y_test*(Y_max - Y_min) + Y_min #dont need 

#Fenics histogram 
u_X = Y_test[:,index].detach().numpy()


#create solution with DNN model
Y_DNN = model(X_test)

#undo normalisation of test outputs:
Y_DNN = Y_DNN*(Y_max - Y_min) + Y_min

#compute DNN evaluation at given x point
u_X_DNN = Y_DNN[:, index].detach().numpy()

# Calculate mean and standard deviation of FEniCSx and DNN distrobutions
mean_FEM = np.mean(u_X)
std_FEM = np.std(u_X)
mean_DNN = np.mean(u_X_DNN)
std_DNN = np.std(u_X_DNN)


print("DNN:")
print("mean ", mean_DNN)
print("std ", std_DNN)
print("FEM:")
print("mean ", mean_FEM)
print("std ", std_FEM)




#plot histogram
plt.figure(figsize=(8, 6))
plt.hist(u_X, bins=50,  alpha=0.6, label='FEM', color='blue')
plt.hist(u_X_DNN, bins=50, alpha=0.6, label='DNN', color='orange')
plt.xlabel('Value of u(x,y)', fontsize= 18)
plt.ylabel('Frequency', fontsize = 18)
plt.title(f'Distribution at [x,y] = {coords[index]}', fontsize = 20)
plt.legend(fontsize = 20)
plt.grid(True)
plt.savefig("UQ for PDE 5.pdf", format="pdf")

plt.show()





#%% Plot solution for u for given k vector
i = 180

x = coords[:,0]
y = coords[:,1]
z = Y_val[i]
z = z*(Y_max - Y_min) + Y_min
z = (( np.array(z.detach().numpy()) ))

print(X_val[i]*(X_max - X_min) + X_min)

# 3d plot
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(x, y, z, c=z)
contour = plt.tricontourf(x, y, z, levels=20, cmap=cm.viridis)
plt.colorbar(contour)
plt.show()

# Success for comparisson with X_val

x = coords[:,0]
y = coords[:,1]
z = model(X_val[i])
z = z*(Y_max - Y_min) + Y_min
z = (( np.array(z.detach().numpy()) ))
print(X_val[i]*(X_max - X_min) + X_min)

# 3d plot
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(x, y, z, c=z)
contour = plt.tricontourf(x, y, z, levels=20, cmap=cm.viridis)
plt.colorbar(contour)
plt.show()



