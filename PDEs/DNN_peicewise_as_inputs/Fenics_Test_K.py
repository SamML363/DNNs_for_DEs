#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:13:23 2025

@author: samuellewis
"""

#Works for a given k need to loop for lots of different k vectors????
# need to change f to be independant of k otherwise kinda pointless as always get same solution?



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

#%% Read in test K vals

def read_txt(txt_file):
    f = open(txt_file, 'r')
    content = f.read()
    lines = content.splitlines()
    dat = [x.split(',') for x in lines]
    dat = np.array(dat,dtype=float)
    f.close()
    return dat[0] if len(dat) == 1 else dat


k_consts = read_txt(r"K.txt")

#%% define the function to call for evaluating function at different coordinates

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



#%% Initialise FE spacesand boundary conditions for problem

#initialise mesh size
mesh_length = 64

#define a the boundaries of the space Omega by creating a mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_length, mesh_length, mesh.CellType.quadrilateral)
input_coords = domain.geometry.x

#  Define the trial function space V
V = fem.functionspace(domain, ("Lagrange", 1))   


# define initial boundary condition uD 
uD = fem.Function(V)
#uD.interpolate(lambda x:  10 + 3*x[0] + 10*np.cos(x[1])) 
uD.interpolate(lambda x:  x[0] *(x[0]-1) * x[1]*(x[1]-1)) 

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



#%% solve u for given k:
    
#initialise storage of solutions
u_sol_vec = np.zeros(((mesh_length+1)**2 ))
    

#k_val = ufl.conditional(ufl.And(x < 0.5, y < 0.5), 1.0, 2.0)
k = fem.Function(V)
k.interpolate(lambda x: k_vals(x, k_consts))


# Define the source term 
f = fem.Constant(domain, default_scalar_type(-6)) # - ufl.div(k * ufl.grad(u_exact))
#f = fem.Function(V)
#f.interpolate(f_vals)

# Define variational form 
a = ufl.inner(ufl.grad(v), k * ufl.grad(u)) * ufl.dx 
L = f * v * ufl.dx

# Solve the linear problem using LU - factorisation
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

u_sol_vec = uh.x.array



#print("solution is uh =:", uh.x.array)


#%% Compute error by moving into the P2 function space to calculate exact solution 



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
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")





#%% plot function

print(k_consts)

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


















