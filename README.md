# MPNNTrans

The run.py file in src is the main file and shows how to use the modules. Unfortunately the modules are fake, and they are stil quite hard-coded. We can try to de-hard-code it such that it gets easier to write new models and specially continuous versions.

## Main.py

This file calculates the L2-error between the GraphSage MPNN and its continuous version. GraphSage is implemented here: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv. We consider no bias, and no root_weight. Then, we read out the weight of GraphSage and compute the continuous MLP. For this, we use the forward function in the local MPNNTrans/src/CGraphSage.py  module. The data is loaded with the RGGDataset class written in  MPNNTrans/src/DataLoader.py.


## Run.py

We now consider a differente MPNN. We calculate the message by a two layer MLP with ReLU activation function. Then, we calculate the mean aggregation node-wise and in the end apply an MLP. For this, we implemented the discrete and continuous model in MPNNTrans/src/TwoLayerGraphSage.py, read in data through MPNNTrans/src/DataLoader.py.
