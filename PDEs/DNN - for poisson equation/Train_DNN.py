#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:39:55 2025

@author: samuellewis
"""


import torch
import scipy.io as sio

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.optim as optim
#from sklearn.model_selection import train_test_split


#%% function


def read_txt(txt_file):
    f = open(txt_file, 'r')
    content = f.read()
    lines = content.splitlines()
    dat = [x.split(',') for x in lines]
    dat = np.array(dat,dtype=float)
    f.close()
    return dat[0] if len(dat) == 1 else dat


#%% Prepare data as tensors


X, Y = read_txt(r"X.txt"), read_txt(r"Y.txt")
X_val, Y_val = read_txt(r"X_val.txt"), read_txt(r"Y_val.txt")

#Y=Y.T

"""
X_test= read_txt(r"X_test.txt")
X_test=X_test.T  

X_test2= read_txt(r"X_test2.txt")
X_test2=X_test2.T
"""


print(X.shape)
print(X_val.shape)
print(Y.shape)
print(Y_val.shape)

"""
print(X_test.shape)
print(X_test2.shape)
"""
  
np.random.seed(2)
torch.manual_seed(2)

#X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, shuffle=True)


# Convert to 2D PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)
"""
X_test = torch.tensor(X_test, dtype=torch.float32)
X_test2 = torch.tensor(X_test2, dtype=torch.float32)
"""

#%% Create the surrogate model
 
# Define the DNN
model = nn.Sequential(
    nn.Linear(2, 20),
    nn.Tanh(),
    nn.Linear(20, 10),
    nn.Tanh(),
    nn.Linear(10, 1),
)
 
# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.00005)
#optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
 
n_epochs = 100000   # number of epochs to run
plot_epoch = 100
batch_size = 70  # size of each batch


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
        loss = loss_fn(Y_pred, Y_batch)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # update parameter
        optimizer.step()
            # print progress
            
    # evaluate accuracy at end of each epoch
    if epoch % plot_epoch == 0:
        model.eval()
        with torch.no_grad():
            Y_pred = model(X)

        with torch.no_grad():
            Y_pred_val = model(X_val)

        mse = loss_fn(Y_pred, Y)
        mse_val = loss_fn(Y_pred_val, Y_val)
        
        mse_vec[index] = mse.item()
        mse_val_vec[index] = mse_val.item()
        x_epoch[index] = epoch
        index+= 1




#%%plot mse vals

plt.close()
plt.semilogy(x_epoch,mse_vec,label="Training")
plt.semilogy(x_epoch,mse_val_vec,label="Validation")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Epochs")
plt.show()

plt.savefig("Poisson DNN MSE.pdf", format="pdf", bbox_inches="tight")


#%%Check optimised theta success - train

x = np.array(X[:,0])
y = np.array(X[:,1])
z = np.array(model(X).detach().numpy()).reshape(x.shape)

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(x, y, z, c=z)

plt.savefig("Poisson DNN model.pdf", format="pdf", bbox_inches="tight")

#%% - validation

x = np.array(X_val[:,0])
y = np.array(X_val[:,1])
z = np.array(model(X_val).detach().numpy()).reshape(x.shape)

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(x, y, z, c=z)

