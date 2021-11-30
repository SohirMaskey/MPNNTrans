#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import matplotlib.pyplot as plt

# In[13]:


# For plotting

# To avoid copying things to GPU memory,
# ideally allocate everything in torch on the GPU
# and avoid non-torch function calls
torch.set_printoptions(precision=10) # Set displayed output precision to 10 digits

from torchquad import enable_cuda # Necessary to enable GPU support
from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS # The available integrators
import torchquad


# In[3]:


from torch_geometric.nn import SAGEConv

enable_cuda() # Use this to enable GPU support


# In[2]:



# In[4]:


N = 19702 #How large shall the graphs become?
skip = 100 #Should we consider all graphs, or only every skip'th
r = 0.2

#graphSage = SAGEConv(graphs[0].num_node_features,1 ,root_weight = False, bias = False)

RGG_data_set_values = []

# In[12]:


eucl_norm = lambda x,y: torch.sqrt(torch.pow(x,2) + torch.pow(y, 2))
def indicator_fct(x, y, distance, center):
    z = (eucl_norm(x-center[0], y- center[1]) < distance)*1
    return z

k = lambda x,y: x*y
g = lambda x,y, distance, center: indicator_fct(x, y, distance, center)*k(x,y)
#h = lambda x: g(x[:,0], x[:,1], 0.5, np.array(pos[50]))
#l = lambda x: indicator_fct(x[:,0],x[:,1], 0.5, np.array(pos[50]))


simp = Simpson()  # Initialize Simpson solver


def graph_l2error(gnn_output, position, length):
    error = 0
    for key, value in position.items():
        h = lambda x: g(x[:,0], x[:,1], r, torch.tensor(value))
        l = lambda x: indicator_fct(x[:,0],x[:,1], r, torch.tensor(value))
        int_of_f = simp.integrate(h,dim=2,N=1000000,integration_domain = [[0,1],[0,1]])
        measure_Ball = simp.integrate(l,dim=2,N=1000000,integration_domain = [[0,1],[0,1]])
        error2 = the_weight*(int_of_f/measure_Ball) - gnn_output[key]
        error = error + torch.pow(error2, 2)
    error = torch.sqrt( (1/length)*error )
    return error


l2_errors = []

with open('RGG_' + str(10101) +'.pickle', 'rb') as data:
    graphs = pickle.load(data)
with open('graph_signal_' + str(10101) +'.pickle', 'rb') as data:
    graph_signals = pickle.load(data)  
with open('pos_' + str(10101) +'.pickle', 'rb') as data:
    pos = pickle.load(data)
RGG = Data(x = torch.tensor(graph_signals), edge_index = graphs.edge_index)
    
graphSage = SAGEConv(graphs.num_node_features,1 ,root_weight = False, bias = False)
 
start = time.time()    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graphSage = graphSage.to(device)
RGG = RGG.to(device)

end = time.time()
print(f"Took {(end-start)* 1000.0:.3f} ms -- " + str(1) + " nodes")    

start = time.time()

forward_value = graphSage.forward(RGG.x, RGG.edge_index)
    
end = time.time()
    
print(f"Took {(end-start)* 1000.0:.3f} ms -- " + str(1) + " nodes")    
    
the_weight = (graphSage.lin_l).weight
    
    
l2_errors.append(graph_l2error(forward_value, pos,10101))
       

start = time.time ()


for j in range(10201,N,skip):

    with open('RGG_' + str(j) +'.pickle', 'rb') as data:
        graphs = pickle.load(data)
    with open('graph_signal_' + str(j) +'.pickle', 'rb') as data:
        graph_signals = pickle.load(data)  
    with open('pos_' + str(j) +'.pickle', 'rb') as data:
        pos = pickle.load(data)
    RGG = Data(x = torch.tensor(graph_signals), edge_index = graphs.edge_index)
    
    RGG = RGG.to(device)
    
    forward_value = graphSage.forward(RGG.x, RGG.edge_index)
    
    #RGG_data_set_values.append(forward_value)
    
    l2_errors.append(graph_l2error(forward_value, pos, len(pos)))

    
end = time.time()
    
print(f"Took {(end-start)* 1000.0:.3f} ms -- " + str(101) + " nodes")    
    

hi = [tens.item() for tens in l2_errors]


he = list(range(0,len(l2_errors)))


fig = plt.figure()
plt.xlabel('Nodes')
plt.ylabel('l2error')
plt.plot(he,hi[:])
fig.savefig('l2error_' + str(N) + 'nodes_' + str(skip) + 'skip' + '.png', dpi=fig.dpi)


with open('l2_errors_' + str(N) + '.pickle', 'wb') as output:
    pickle.dump(l2_errors, output)


# In[3]:


he = list(range(0,N, skip))
fig = plt.figure()
plt.xlabel('Nodes')
plt.ylabel('l2error')
plt.plot(he,hi[:])
fig.savefig('l2error_' + str(N) + 'nodes_' + str(skip) + 'skip' + '.png', dpi=fig.dpi)


# In[4]:


slope, intercept = np.polyfit(np.log(he[1:]), np.log(hi[1:]), 1)
print(slope)
plt.loglog(he[1:], hi[1:], '--')
fig.savefig('loglogerror_' + str(N) + 'nodes_' + str(skip) + 'skip' + '.png', dpi=fig.dpi)


# In[ ]:




