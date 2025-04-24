#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:13:23 2025

@author: samuellewis
"""



#%% import items
from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
import numpy as np
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import pyvista
from dolfinx import plot
import pyvistaqt
import os

import time
#%% define the function to call for evaluating peicewise function at different coordinates

#Define k as a peicewise function
def k_vals(x, peicewise_consts):
    values = np.zeros(x.shape[1])
    mask = (x[0] <= 0.34) & (x[1] <= 0.34)
    values[mask] = peicewise_consts[0]
    
    mask = (0.34< x[0]) & (x[0] <= 0.67) & (x[1] <= 0.34)
    values[mask] = peicewise_consts[1]
    
    mask = (0.67< x[0]) & (x[1] <= 0.34)
    values[mask] = peicewise_consts[2]
    
    mask = ( x[0] <= 0.34) & (0.34< x[1] ) & ( x[1] <= 0.67)
    values[mask] = peicewise_consts[3]
    
    mask = ( 0.34 < x[0] ) & ( x[0] <= 0.67) & (0.34< x[1] ) & ( x[1] <= 0.67)
    values[mask] = peicewise_consts[4]
    
    mask = ( 0.67 < x[0]) & (0.34< x[1] ) & ( x[1] <= 0.67)
    values[mask] = peicewise_consts[5]
    
    mask = ( x[0] <= 0.34) & (0.67 < x[1])
    values[mask] = peicewise_consts[6]
    
    mask = ( 0.34 < x[0] ) & (  x[0] <= 0.67) & (0.67 < x[1])
    values[mask] = peicewise_consts[7]
    
    mask = ( 0.67 < x[0] ) & (0.67 < x[1])
    values[mask] = peicewise_consts[8]
    
    return values

def f_vals(x):
    return np.exp(1/(x[0]+x[1]))


#%% Initialise the different vectors for k:
    
#define number of data points for DNN (training + validation)
data_size = 18000

#create random peicewise constants
k_consts = np.random.randint(low = 1,high = 10, size = (data_size,9))



#%% Initialise FE spacesand boundary conditions for problem

#initialise mesh size
mesh_length = 32

#define a the boundaries of the space Omega by creating a mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_length, mesh_length, mesh.CellType.quadrilateral)
input_coords = domain.geometry.x

#  Define the trial function space V
V = fem.functionspace(domain, ("Lagrange", 1))   


# define initial boundary condition uD 
uD = fem.Function(V)
uD.interpolate(lambda x:  x[0] *(x[0]-1) * x[1]*(x[1]-1)) # note that for the unit square this is the same as uD = 0

#define boundaries of mesh for BC to be applied on :
tdim = domain.topology.dim # topological dimension of domain e.g 3D
fdim = tdim - 1 # dimension of facet e.g the edge of a 3D shape is 2D
domain.topology.create_connectivity(fdim, tdim) #creates a connectivity map between facets and cells to ensure it is possible to identify which boundary facet belongs to which element
boundary_facets = mesh.exterior_facet_indices(domain.topology) # This collects idices of which facets are boundary facets - required for applying boundary conditions

#Enforce boundary condition on required DoF's
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets) #numpy array of indices of degrees of freedom at boundery of mesh
bc = fem.dirichletbc(uD, boundary_dofs) #Applies derichlet BC uD 

#define the trial and test functions 
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

#define exact value of u
x, y = ufl.SpatialCoordinate(domain)
u_exact = x*(x-1)*y*(y-1)



#%% Loop to solve for u for given k:
    
start_time = time.time()
    
#initialise storage of solutions
u_sol_vec = np.zeros((data_size, (mesh_length+1)**2 ))
    
for i in range(k_consts.shape[0]):

    #k_val = ufl.conditional(ufl.And(x < 0.5, y < 0.5), 1.0, 2.0)
    k = fem.Function(V)
    k.interpolate(lambda x: k_vals(x, k_consts[i]))
    
    
    # Define the source term 
    f = fem.Constant(domain, default_scalar_type(-6)) #option for if you want a known true solution: - ufl.div(k * ufl.grad(u_exact))
  
    
    # Define variational form 
    a = ufl.inner(ufl.grad(v), k * ufl.grad(u)) * ufl.dx 
    L = f * v * ufl.dx
    
    # Solve the linear problem using LU - factorisation
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    
    u_sol_vec[i] = uh.x.array
    


    #print("solution is uh =:", uh.x.array)

end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.6f} seconds")

#%% Compute error by moving into the P2 function space to calculate exact solution - (for this to be a valid test the source term must be changed to the other option)

"""

#same as above but with 2nd order lagrange polynomials 
V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x:  x[0] *(x[0]-1)*x[1]*(x[1]-1)) 


#calculate L2 norm of error uD - uh
L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
# evaluate the integral that is L2_error and return a scaler
error_local = fem.assemble_scalar(L2_error)
# if running in parrallel on multiple processors this next line is required to combine all errors
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

#calculate the maximum error at any DoF
error_max = np.max(np.abs(uD.x.array-uh.x.array))
#uh.X.array is the numerical solution stored in finite element function uh
# Only print the error on one process
if domain.comm.rank == 0:
    #rank 0 ensure only one processor prints error if running accross multiple processors 
    print(i)
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

"""



#%% plot function

#print(k_consts[-1])

pyvista.set_jupyter_backend(None) # ensure not using jupiter backend allowing it to work in python


#extract mesh data from function space V (which uh is a solution inthis space )
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
#convert to pyvista grid
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
#store the FE solution as Pyvista point data 
u_grid.point_data["uh"] = uh.x.array.real #.real used incase of colpex numbers 
#tell Pyvista to use "u" as active scalar visualisation: 
u_grid.set_active_scalars("uh")

u_plotter = pyvistaqt.BackgroundPlotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show_bounds(grid = True)
u_plotter.show()



#%% write to file for use with pytorch DNN training

# if using higher function space then L1 then will need to interpolate uh array onto lower order space in order to correspond to mesh points:
    #uh = fem.functionspace(domain, ("Lagrange", 1))  
    #uh_low.interpolate(uh)
    # Y = uh_low.x.array



#write input and output data for training DNN:
X = k_consts
Y = u_sol_vec
xy_coords = input_coords [:, :2]
indices = np.random.permutation(len(Y))
X = X[indices]
Y=Y[indices]


#break into train val and test data
X_train = X[:7000]
X_val = X[7000:9000]
X_test = X[9000:]

Y_train = Y[:7000]
Y_val = Y[7000:9000]
Y_test = Y[9000:]





#%%

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
#script_directory = os.path.dirname(os.path.abspath(__file__))

# File paths (in the same directory as the script)
x_file = os.path.join(script_directory, "X.txt")
y_file = os.path.join(script_directory, "Y.txt")
x_val_file = os.path.join(script_directory, "X_val.txt")
y_val_file = os.path.join(script_directory, "Y_val.txt")
x_test_file = os.path.join(script_directory, "X_test.txt")
y_test_file = os.path.join(script_directory, "Y_test.txt")

#write coords from file so can be recovered later
coords_file = os.path.join(script_directory, "coords.txt")

# Write the arrays to text files
write_txt(X_train, 'X.txt')    # Training features
write_txt(Y_train, 'Y.txt')    # Training targets
write_txt(X_val, 'X_val.txt')  # Validation features
write_txt(Y_val, 'Y_val.txt')  # Validation targets
write_txt(X_test, 'X_test.txt')    # Test features
write_txt(Y_test, 'Y_test.txt')    # Test targets


write_txt(xy_coords, 'coords.txt')  # Validation targets














