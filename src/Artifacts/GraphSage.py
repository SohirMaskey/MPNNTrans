#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle #for saving and loading the dataset

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import time


# In[2]:


import random
from torch_geometric.utils import from_networkx


# In[3]:


from torch_geometric.nn import SAGEConv


# In[4]:


N = 11000
split = 1000
r = 0.5


# In[5]:


with open('RGG_dataset.pickle', 'rb') as data:
    graphs = pickle.load(data)


# In[6]:


graphSage = SAGEConv(graphs[0].num_node_features,1 ,root_weight = False, bias = False)


# In[7]:


RGG_data_set = []


# In[8]:


with open('graph_signals.pickle', 'rb') as data:
    graph_signals = pickle.load(data)


# In[9]:


for g, s in zip(graphs, graph_signals):
    RGG_data_set.append(Data(x = torch.tensor(s), edge_index = g.edge_index))


# In[10]:


RGG_data_set_values = [graphSage.forward(data.x, data.edge_index) for data in RGG_data_set]


# In[11]:


the_weight = (graphSage.lin_l).weight


# In[12]:


eucl_norm = lambda x,y: np.sqrt(x**2 + y**2)
def indicator_fct(x, y, distance, center):
    z = (eucl_norm(x-center[0], y- center[1]) < distance)*1
    return z

k = lambda x,y: x*y
g = lambda x,y, distance, center: indicator_fct(x, y, distance, center)*k(x,y)
#h = lambda x: g(x[:,0], x[:,1], 0.5, np.array(pos[50]))
#l = lambda x: indicator_fct(x[:,0],x[:,1], 0.5, np.array(pos[50]))


# In[13]:


# For plotting
import matplotlib.pyplot as plt

# To avoid copying things to GPU memory,
# ideally allocate everything in torch on the GPU
# and avoid non-torch function calls
torch.set_printoptions(precision=10) # Set displayed output precision to 10 digits

from torchquad import enable_cuda # Necessary to enable GPU support
from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS # The available integrators
import torchquad


# In[14]:


simp = Simpson()  # Initialize Simpson solver


# In[15]:


with open('pos.pickle', 'rb') as data:
    pos = pickle.load(data)


# In[16]:


def graph_l2error(gnn_output, position):
    error = 0
    for key, value in position.items():
        h = lambda x: g(x[:,0], x[:,1], r, np.array(value))
        l = lambda x: indicator_fct(x[:,0],x[:,1], r, np.array(value))
        int_of_f = simp.integrate(h,dim=2,N=1000000,integration_domain = [[0,1],[0,1]])
        measure_Ball = simp.integrate(l,dim=2,N=1000000,integration_domain = [[0,1],[0,1]])
        error2 = the_weight*(int_of_f/measure_Ball) - gnn_output[key]
        error = error + error2**2
    error = torch.sqrt( 1/len(RGG_data_set_values)*error )
    return error


# In[17]:


l2_errors = []


# In[18]:


for i in range(0,len(pos)):
    l2_errors.append(graph_l2error(RGG_data_set_values[i], pos[i]))


# In[19]:


hi = [tens.item() for tens in l2_errors]


# In[20]:


he = list(range(0,len(pos)))


# In[21]:


fig = plt.figure()
plt.xlabel('Nodes')
plt.ylabel('l2error')
plt.plot(he,hi[:])
fig.savefig('l2error_1000N.png', dpi=fig.dpi)


# In[22]:


with open('l2_errors2.pickle', 'wb') as output:
    pickle.dump(l2_errors, output)

