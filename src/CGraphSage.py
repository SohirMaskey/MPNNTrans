#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch


# For plotting

# To avoid copying things to GPU memory,
# ideally allocate everything in torch on the GPU
# and avoid non-torch function calls
torch.set_printoptions(precision=10) # Set displayed output precision to 10 digits

from torchquad import enable_cuda # Necessary to enable GPU support
from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS # The available integrators
import torchquad


enable_cuda() # Use this to enable GPU support


eucl_norm = lambda x,y: torch.sqrt(torch.pow(x,2) + torch.pow(y, 2))

def indicator_fct(x, y, distance, center):
    z = (eucl_norm(x-center[0], y- center[1]) < distance)*1
    return z

k = lambda x,y: x*y

r = 0.2

simp = Simpson()  # Initialize Simpson solver

def forward(positions, weight, fct=k):
    """
    pipeline should be as follows in the main:
    1. load model and its weights
    2. load graph data and its positions
    3. apply forward to return a torch.tensor with the c-GraphSage output for every position
    """
    k = fct
    g = lambda x,y, distance, center: indicator_fct(x, y, distance, center)*fct(x,y)
    #r = 0.2
    values = []
    the_weight = weight
    
    for i in range(0, len(positions)):
        h = lambda x: g(x[:,0], x[:,1], 0.2, positions[i])
        l = lambda x: indicator_fct(x[:,0],x[:,1], 0.2, positions[i])
        
        int_of_f = simp.integrate(h,dim=2,N=1000000,integration_domain = [[0,1],[0,1]])
        measure_Ball = simp.integrate(l,dim=2,N=1000000,integration_domain = [[0,1],[0,1]])
        
        values.append([torch.tensor(the_weight*(int_of_f/measure_Ball))])
        print(i)
    return torch.tensor(values)
        
        
    """
    for key, value in position.items():
        h = lambda x: g(x[:,0], x[:,1], r, torch.tensor(value))
        l = lambda x: indicator_fct(x[:,0],x[:,1], r, torch.tensor(value))
        int_of_f = simp.integrate(h,dim=2,N=1000000,integration_domain = [[0,1],[0,1]])
        measure_Ball = simp.integrate(l,dim=2,N=1000000,integration_domain = [[0,1],[0,1]])
        error2 = the_weight*(int_of_f/measure_Ball) - gnn_output[key]
        error = error + torch.pow(error2, 2)
    error = torch.sqrt( (1/length)*error )
    return error

    """