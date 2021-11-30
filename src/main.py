import pickle 

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import time


# In[2]:


import random
from torch_geometric.utils import from_networkx

import matplotlib.pyplot as plt

"""
torch.set_printoptions(precision=10) # Set displayed output precision to 10 digits

from torchquad import enable_cuda # Necessary to enable GPU support
from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS # The available integrators
import torchquad


"""


from torch_geometric.nn import SAGEConv

###Own Moduls
import sys
sys.path.insert(1,'../src')
from DataLoader import RGGDataset
from CGraphSage import forward

def main():
    DL = RGGDataset(root = "../input", size = 1002)
    model = torch.load("../models/graphSageOneLayer.pt")
    the_weight = (model.lin_l).weight
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    L2Errors = []
    for i in range(1, 1002, 100):
        pos = DL.get(i).pos
        cOutput = forward(positions = pos, weight = the_weight)
        data = DL.get(i)
        data = data.to(device)
        nodeErrors = cOutput - model.forward(data.x, data.edge_index)
        L2Error = torch.sqrt(1/len(nodeErrors)*torch.sum(torch.pow(nodeErrors,2)))
        L2Errors.append(L2Error)
    return L2Errors


errors = main()

err = [tens.item() for tens in errors]

with open('../output/l2Errors' + str(1002) + 'Nodes' + '.pickle', 'wb') as output:
    pickle.dump(err, output)
    
    
xAxis = list(range(0,1002, 100))
fig = plt.figure()
plt.xlabel('Nodes')
plt.ylabel('l2error')
plt.plot(xAxis,err[:])
fig.savefig('../output/l2Error' + str(1002) + 'Nodes.png', dpi=fig.dpi)



slope, intercept = np.polyfit(np.log(xAxis[1:]), np.log(err[1:]), 1)
print(slope)
plt.loglog(xAxis[1:], err[1:], '--')
fig.savefig('../output/Logl2Error' + str(1002) + 'Nodes.png', dpi=fig.dpi)