import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import time

import random
from torch_geometric.utils import from_networkx

import matplotlib.pyplot as plt

from torch_geometric.nn import SAGEConv

torch.set_printoptions(precision=10) # Set displayed output precision to 10 digits

from torchquad import enable_cuda # Necessary to enable GPU support
from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS # The available integrators
import torchquad

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
        
        
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = torch.rand(output_size, input_size) - 0.5
        self.bias = torch.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward(self, input_fct):
        self.input_fct = input_fct
        fct = lambda x,y: ((self.input_fct)(x,y)*(self.weights)).sum(-1) + self.bias
        self.output = fct
        return self.output
    
    def set_weights(self, tensor):
        self.weights = tensor
    
    def set_bias(self, tensor):
        self.bias = tensor

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    """
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    """
    
eucl_norm = lambda x,y: torch.sqrt(torch.pow(x,2) + torch.pow(y, 2))

def indicator_fct(x, y, distance, center):
    z = (eucl_norm(x-center[0], y- center[1]) < distance)*1
    return z

#k = lambda x,y: x*y

simp = Simpson()  # Initialize Simpson solver


class meanIntegration(Layer):
    def __init__(self, input_size, output_size):
    #self.weights = torch.rand(output_size, input_size) - 0.5
    #self.bias = torch.rand(1, output_size) - 0.5
        self.weights = torch.rand(output_size, input_size) - 0.5
    
    # returns output for a given input
    def forward(self, input_fct, radius):
        self.input_fct = input_fct
        self.radius = radius

        g = lambda x,y, distance, center: indicator_fct(x, y, distance, center)*self.input_fct(x,y)
            
        #for i in range(0, len(positions)):
         #   h = lambda x: g(x[:,0], x[:,1], r, positions[i])
         #   l = lambda x: indicator_fct(x[:,0],x[:,1], r, positions[i])
        #
        
        def output_fct(position):
            h = lambda x: g(x[:,0], x[:,1], radius, position)
            l = lambda x: indicator_fct(x[:,0],x[:,1], radius, position)
        
            int_of_f = simp.integrate(h,dim=2,N=1000000,integration_domain = [[0,1],[0,1]])
            measure_Ball = simp.integrate(l,dim=2,N=1000000,integration_domain = [[0,1],[0,1]])
        
            return torch.tensor([[int_of_f/measure_Ball]])
        
        return output_fct
        #values.append(torch.tensor(the_weight*(int_of_f/measure_Ball)))
        #return torch.tensor(values)