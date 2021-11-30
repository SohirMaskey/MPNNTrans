import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import time

import random
from torch_geometric.utils import from_networkx

import matplotlib.pyplot as plt

from torch_geometric.nn import SAGEConv

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

import sys
sys.path.insert(1,'../src')
from DataLoader import RGGDataset
from TwoLayerGraphSage import GCN, cGCN

DL = RGGDataset(root = '../input')

model = GCN()
model.load_state_dict(torch.load( '../models/GCNTwoLayersGraphSage'))

cGCNN = cGCN()
model.load_state_dict(torch.load( '../models/cGCNTwoLayersGraphSage'))

fct = lambda x: x[:,0]*x[:,1]
L2Errors = []
start = time.time()
for i in range(1, 502, 100):
    data = DL.get(i) 
    pos = data.pos
    #pos = pos.to(device)
    cfct = cGCNN.forward(fct)
        
    b = torch.empty(( 0 ))
    for i, w in enumerate(pos):
        b = torch.cat((b, cfct(w)), 0)
        #d = torch.reshape(b,(4, len(x)))
        
    #data = DL.get(i)
    #data = data.to(device)
    nodeErrors = b - model.forward(data)
    L2Error = torch.sqrt(1/len(nodeErrors)*torch.sum(torch.pow(nodeErrors,2)))
    L2Errors.append(L2Error)
end = time.time()
print(f"Took {(end-start)* 1000.0:.3f} ms")

xAxis = list(range(0,502, 100))
fig = plt.figure()
plt.xlabel('Nodes')
plt.ylabel('l2error')
plt.plot(xAxis,err[:])
fig.savefig('../output/2LayerGraphSagel2Error' + str(502) + 'Nodes.png', dpi=fig.dpi)

slope, intercept = np.polyfit(np.log(xAxis[1:]), np.log(err[1:]), 1)
print(slope)
plt.loglog(xAxis[1:], err[1:], '--')
fig.savefig('../output/Log2LayerGraphSagel2Error' + str(502) + 'Nodes.png', dpi=fig.dpi)