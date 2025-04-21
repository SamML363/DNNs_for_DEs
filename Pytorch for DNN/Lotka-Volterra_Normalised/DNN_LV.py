#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:01:38 2025

@author: samuellewis
"""

import torch
import scipy.io as sio

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.optim as optim
#from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata


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

#%%
 

# Define the DNN
model = nn.Sequential(
    nn.Linear(2, 20),
    nn.Tanh(),
    nn.Linear(20, 10),
    nn.Tanh(),
    nn.Linear(10, 2),
)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
n_epochs = 5000  # number of epochs to run
plot_epoch = 100
batch_size = 300  # size of each batch


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


"""
#Predict on testing data        
Y_test = model(X_test)

Y_test=Y_test.detach().numpy()
Y_test2 = model(X_test2)
Y_test2=Y_test2.detach().numpy()
sio.savemat('Y_pred.mat', {'YTest':Y_test,'YTest2':Y_test2})

"""



#%%plot mse vals

plt.close()
plt.semilogy(x_epoch,mse_vec,label="Training")
plt.semilogy(x_epoch,mse_val_vec,label="Validation")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Epochs")
plt.show()
plt.savefig("LV_MSE.pdf", format="pdf", bbox_inches="tight")

#plt.savefig("mse.png")



#%%Check optimised theta success

g = np.array(X[:,0])
d = np.array(X[:,1])

QoI = np.array(model(X).detach().numpy())
QoI_x, QoI_y = QoI[:,0] , QoI[:,1]

#%%
#inverse normalisation
g = x1_min + (g -L)*(x1_max - x1_min) / (U-L)
d = x2_min + (d -L)*(x2_max - x2_min) / (U-L)

#inverse normalisation
QoI_x = y1_min + (QoI_x -L)*(y1_max - y1_min) / (U-L)
QoI_y = y2_min + (QoI_y -L)*(y2_max - y2_min) / (U-L)

#%%

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(g, d, QoI_x, c=QoI_x)
#plt.savefig("LV_QoI_x_DNN.pdf", format="pdf", bbox_inches="tight")

#%%
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(g, d, QoI_y, c=QoI_y)
#plt.savefig("LV_QoI_y_DNN.pdf", format="pdf", bbox_inches="tight")

#%% contor plot of errors vs true solution:
    
true_x = Y[:, 0].detach().cpu().numpy().copy()  
true_y = Y[:, 1].detach().cpu().numpy().copy()
    
true_x = y1_min + (true_x -L)*(y1_max - y1_min) / (U-L)
true_y = y2_min + (true_y -L)*(y2_max - y2_min) / (U-L)

error_x = np.abs(true_x - QoI_x)
error_y = np.abs(true_y - QoI_y)


x_coord = g
y_coord = d

grid_x, grid_y = np.mgrid[min(x_coord):max(x_coord):1000j, min(y_coord):max(y_coord):1000j]

grid_z = griddata((x_coord, y_coord), error_x, (grid_x, grid_y), method='linear')

# Create contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='coolwarm')
plt.colorbar(contour, label="Error Magnitude")
plt.xlabel(r"$\gamma$", fontsize = 20)
plt.ylabel(r"$\delta$", fontsize = 20)
plt.title("Error for poulation x", fontsize = 20)
plt.savefig("LV_Error_x.pdf", format="pdf", bbox_inches="tight")
plt.show()


grid_z2 = griddata((x_coord, y_coord), error_y, (grid_x, grid_y), method='linear')

# Create contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(grid_x, grid_y, grid_z2, levels=100, cmap='coolwarm')
plt.colorbar(contour, label="Error Magnitude")
plt.xlabel(r"$\gamma$", fontsize = 20)
plt.ylabel(r"$\delta$", fontsize = 20)
plt.title("Error for poulation y", fontsize = 20)
plt.savefig("LV_Error_y.pdf", format="pdf", bbox_inches="tight")
plt.show()

