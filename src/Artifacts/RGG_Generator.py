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


graphs = []

graph_signals = []

positions = []


# In[4]:


f = lambda x:x[0]*x[1] #the function we use to sample the node signals, can be changed to any


# In[5]:


N = 20002 #How large shall the graphs become?
skip = 200 #Should we consider all graphs, or only every skip'th
r = 0.5


# In[6]:


start = time.time()


# In[7]:


for j in range(1,N,skip):
    graph_signal = []
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(j)}
    for i in range(j):
        graph_signal.append([f(pos[i])])
    graphs.append(nx.random_geometric_graph(j, r, pos = pos))
    graph_signals.append(graph_signal)
    positions.append(pos)


# In[8]:


graphs = [from_networkx(g) for g in graphs]


# In[9]:


end = time.time()
print(f"Took {(end-start)* 1000.0:.3f} ms")


# In[10]:


with open('RGG_dataset_new.pickle', 'wb') as output:
    pickle.dump(graphs, output)


# In[11]:


with open('graph_signals_new.pickle', 'wb') as output:
    pickle.dump(graph_signals, output)


# In[12]:


with open('pos.pickle_new', 'wb') as output:
    pickle.dump(positions, output)


# In[ ]:




