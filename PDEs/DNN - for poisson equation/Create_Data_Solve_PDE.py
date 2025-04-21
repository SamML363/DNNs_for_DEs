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



#%% solve equation

#define a the boundaries of the space Omega by creating a mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.quadrilateral)
input_coords = domain.geometry.x

#  Define the trial function space V
V = fem.functionspace(domain, ("Lagrange", 1))   

# define initial boundary condition uD 
uD = fem.Function(V)
uD.interpolate(lambda x:  1+ x[0]**2 + 2* x[1]**2)

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
#u_exact = x*(x-1)*y*(y-1)
#k = x**2 + y**2 + 1

#Define k
"""
def k_vals(x):
    values = np.ones(x.shape[1])
    mask = (x[0] < 0.33) & (x[1] < 0.33)
    values[~mask] = 2.0
    mask = (x[0] < 0.66) & (x[1] < 0.66)
    values[~mask] = 3.0
    return values
"""

#k_val = ufl.conditional(ufl.And(x < 0.5, y < 0.5), 1.0, 2.0)
#k = fem.Function(V)
#k.interpolate(k_vals)

# Define the source term 
#f = - ufl.inner(ufl.grad(k),ufl.grad(u_exact))
#f = - ufl.div(k * ufl.grad(u_exact))
f = fem.Constant(domain, default_scalar_type(-6))

# Define variational form 
#a = ufl.inner(ufl.grad(v), k * ufl.grad(u)) * ufl.dx 
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Solve the linear problem using LU - factorisation
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

#print("solution is uh =:", uh.x.array)



#%% Compute error by moving into the P2 function space to calculate exact solution 

#same as above but with 2nd order lagrange polynomials 
V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x:  1+ x[0]**2 + 2* x[1]**2)


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
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")





#%% plot function

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


X = input_coords [:, :2]
Y = uh.x.array
indices = np.random.permutation(len(Y))
X = X[indices]
Y=Y[indices]
X_train = X[:700]
X_val = X[700:]
Y_train = Y[:700]
Y_val = Y[700:]




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

# Write the arrays to text files
write_txt(X_train, 'X.txt')    # Training features
write_txt(Y_train, 'Y.txt')    # Training targets
write_txt(X_val, 'X_val.txt')  # Validation features
write_txt(Y_val, 'Y_val.txt')  # Validation targets















