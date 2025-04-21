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
def Create_Model():
 model = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, 5),
            nn.Tanh(),
            nn.Linear(5, 1)
            )
 return model
 
# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
 
n_epochs = 5000  # number of epochs to run
plot_epoch = 100
batch_size = 3000  # size of each batch


#Save for plotting
mse_vec= np.zeros((int(n_epochs/plot_epoch),4))
mse_val_vec= np.zeros((int(n_epochs/plot_epoch),4))
x_epoch= np.zeros((int(n_epochs/plot_epoch),4))


for opt_num in range(4):
    index = 0
    
    model = Create_Model()
    if opt_num == 0:
        optimizer = optim.SGD(model.parameters(), lr=0.1)
    elif opt_num == 1:
        optimizer = optim.SGD(model.parameters(), lr=0.05,momentum=0.9)
    elif opt_num == 2:
        optimizer = optim.Adadelta(model.parameters())# lr=1, rho= 0.9, eps = 1e-06)
    elif opt_num == 3:
        optimizer = optim.Adam(model.parameters(), lr=0.006, betas= (0.9, 0.99), eps = 1e-08)
       
    
    
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
            
            mse_vec[index][opt_num] = mse.item()
            mse_val_vec[index][opt_num] = mse_val.item()
            x_epoch[index][opt_num] = epoch
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
#plt.semilogy(x_epoch,mse_vec,label="Training")
plt.semilogy(x_epoch[:,0],mse_val_vec[:,0],label="SGD")
plt.semilogy(x_epoch[:,1],mse_val_vec[:,1],label="SGD+momentum")
plt.semilogy(x_epoch[:,2],mse_val_vec[:,2],label="AdaDelta")
plt.semilogy(x_epoch[:,3],mse_val_vec[:,3],label="Adam")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Epochs")
plt.show()

plt.savefig("comparison.png")

#%%Check optimised theta success

x = np.array(X[:,0])
y = np.array(X[:,1])
z = np.array(model(X).detach().numpy()).reshape(30000,)

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(x, y, z, c=z)