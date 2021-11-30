import os.path as osp
import pickle

import torch
from torch_geometric.data import Dataset, download_url

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
import os

class RGGDataset(Dataset):
    
    def __init__(self, root, fct = lambda x: x[0]*x[1] , transform=None, pre_transform=None, size=1002, skip=100):
        """
        root = where the dataset should be stored. 
        This folder is split into raw_dir (downloaded dataset) and processed_dir (processed data)

        """
        #super(RGGDataset, self).__init__(root, transform, pre_transform)
        self.fct = fct #this is the function that we use for sampling things
        self.size = size
        self.skip = skip
        self.root = root
    
    size = 1002
    skip = 100
    
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return [f'RGG_{i}.pickle' for i in range(1, self.size, self.skip)]
    
    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        
        return [f'data_{i}.pt' for i in range(1, self.size, self.skip)]
    
    def download(self):
        #we dont really download anything graph_signal_10001.pickle
        pass

    def process(self):    
        for i in range(1, self.size, self.skip): #need to give the length somehow
            with open(f'../input/raw/RGG_{i}.pickle', 'rb') as data:
                graph = pickle.load(data)
            
            try:
                with open(f'../input/raw/graph_signal_{i}.pickle', 'rb') as data:
                    graph_signal = pickle.load(data)
            except:
                graph_signal = torch.ones(i)
            
            #with open(f'../input/graph_signal_{i}.pickle', 'rb') as data:
            #    graph_signal = pickle.load(data)
                
            #with open(f'../input/pos_{i}.pickle', 'rb') as data:
            #    self.pos = pickle.load(data)
            # Create data object
            graph.x = torch.tensor(graph_signal)
            #data.pos = self.pos #Every graph data object gets a position attribute
            torch.save(graph, 
                os.path.join(self.processed_dir, 
                    f'data_{i}.pt'))

    def len(self):
        return len(self.size)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
    
    """
    I do not write a function to get and set node features and edge features because the Data objects 
    already have it, so that one can iterate through the data object then.
    """
    
if __name__ == "__main__":
    RGG10002 = RGGDataset(root = "../input", size = 10002
                         )
    RGG10002.process()