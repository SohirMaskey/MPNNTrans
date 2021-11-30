#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pickle #for saving and loading the dataset

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import time


# In[31]:


import random
from torch_geometric.utils import from_networkx


# In[32]:


graphs = []

graph_signals = []


# In[33]:


f = lambda x:x[0]*x[1] #the function we use to sample the node signals, can be changed to any


# In[34]:


N = 1000 #How large shall the graphs become?
skip = 1 #Should we consider all graphs, or only every skip'th
r = 0.5


# In[35]:


start = time.time()


# In[36]:


for j in range(1,N,skip):
    graph_signal = []
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(j)}
    for i in range(j):
        graph_signal.append([f(pos[i])])
    graphs.append(nx.random_geometric_graph(j, r, pos = pos))
    graph_signals.append(graph_signal)


# In[37]:


graphs = [from_networkx(g) for g in graphs]


# In[38]:


end = time.time()
print(f"Took {(end-start)* 1000.0:.3f} ms")


# In[39]:


with open('RGG_dataset.pickle', 'wb') as output:
    pickle.dump(graphs, output)


# In[40]:


with open('graph_signals.pickle', 'wb') as output:
    pickle.dump(graph_signals, output)


# In[41]:


with open('pos.pickle', 'wb') as output:
    pickle.dump(pos, output)


# In[ ]:




