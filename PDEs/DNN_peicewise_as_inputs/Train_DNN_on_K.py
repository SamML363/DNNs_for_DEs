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

#Y=Y.T


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
Y_test = (Y_test-Y_min)/(Y_max-Y_min)




#%% Create the surrogate model
 

"""
# Define the DNN
model = nn.Sequential(
    nn.Linear(X.shape[1], 64),
    nn.Tanh(),
    nn.Linear(64, 128),
    nn.Tanh(),
    nn.Linear(128, 256),
    nn.Tanh(),
    nn.Linear(256, 512),
    nn.Tanh(),
    nn.Linear(512, Y.shape[1])
)
"""


"""

# Define the DNN
model = nn.Sequential(
    nn.Linear(X.shape[1], 100),
    nn.Tanh(),
    nn.Linear(100, Y.shape[1])
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
    nn.Linear(X.shape[1], 64),
    nn.Sigmoid(),
    nn.Linear(64, 128),
    nn.Sigmoid(),
    nn.Linear(128, 256),
    nn.Sigmoid(),
    nn.Linear(256, 512),
    nn.Sigmoid(),
    nn.Linear(512, Y.shape[1])
)


 
# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0006)
#optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
 
n_epochs = 3000   # number of epochs to run
plot_epoch = 100
batch_size = int(X.shape[0] /10) # size of each batch
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
plt.semilogy(x_epoch,mse_vec,label="Training MSE")
plt.semilogy(x_epoch,mse_val_vec,label="Validation MSE")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Epochs")
plt.title("Learning curve for DNN" )
plt.show()

#plt.savefig("k DNN learning curve simple network.pdf", format="pdf")




#%% create random array not in X for testing

X_train =  {tuple(row.tolist()) for row in X}

while True:
    new_X = tuple(np.random.randint(low = 1,high = 100, size = (9)))  # Generate a random vector
    if new_X not in X_train:  # Check if it's unique
        break

new_X = torch.tensor(new_X)
new_X =  (new_X-X_min)/(X_max-X_min)

#%%Check optimised theta success for camparisson with pyvista




x = coords[:,0]
y = coords[:,1]
z = model(new_X)
z = z*(Y_max - Y_min) + Y_min
z = (( np.array(z.detach().numpy()) ))


#contor plot
fig = plt.figure(figsize=(10, 8))
contour = plt.tricontourf(x, y, z, levels=20, cmap=cm.viridis)
plt.colorbar(contour)


#%% plot true solution for reference
i = 5

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
#plt.colorbar(contour)
plt.title("DNN approximation")
plt.savefig("k_DNN_Approx_sol.pdf", format="pdf")
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
#plt.colorbar(contour)
plt.title("Exact Solution")
plt.savefig("k_DNN_Exact_sol.pdf", format="pdf")
plt.show()


#%% write testing k vals to script

# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.realpath(__file__))

# File paths (in the same directory as the script)
k_file = os.path.join(script_directory, "K.txt")

#undo normalisation
new_X = new_X*(X_max - X_min) + X_min

#switch to numpy array
new_X = (( np.array(new_X.detach().numpy()) ))

# Write the arrays to text files
write_txt(new_X, 'K.txt')    # Training features




#%% testing 


test_diff = loss_fn(Y_val[1], model(X_val[1]) )
print("mse = ", test_diff)


