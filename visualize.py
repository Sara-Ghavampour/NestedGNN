
#python run_tu.py --model NestedGCN --h 3 --hiddens 128  --layers 4 --node_label spd --use_rd --data MUTAG

import torch
import random
import math
import pdb
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as ssp
from scipy import linalg
from scipy.linalg import inv, eig, eigh
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_scatter import scatter_min
from batch import Batch
from collections import defaultdict
from kernel.gcn import NestedGCN
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import scipy.sparse as sp
# from bokeh.palettes import Category20_20, Category20b_20, Accent8
from matplotlib import collections  


from dataloader import DataLoader 
from kernel.datasets import get_dataset

# num_subgraphs
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
model = torch.load('./models/nestedgcn1.pt')


dataset = get_dataset(
            'MUTAG', 
            True, 
            3, 
            'spd', 
            False, 
            None, 
            False, 
            False, 
            None, 
        )
print(dataset[3])
test_loader = DataLoader(dataset, 128, shuffle=False)

loss =0

dataset_embeddings=torch.empty((0,2), dtype=torch.float32)
labels=torch.empty((0), dtype=torch.int32)
class_activation_mappings=torch.empty((0,32,2), dtype=torch.float32)
for data in test_loader:
        data = data.to(device)
        data = data.to(device)
        labels=torch.cat((labels,data.y.view(-1)),axis=0)
        with torch.no_grad():
            out,features ,cam= model(data,return_features=True)
            print('features: ',features.shape)
            print('Class Activation Mapping (CAM): ',cam.shape)
            dataset_embeddings=torch.cat((dataset_embeddings,features),axis=0)
            class_activation_mappings=torch.cat((class_activation_mappings,cam),axis=0)
        loss += F.nll_loss(features, data.y.view(-1), reduction='sum').item()

print('dataset_embeddings : ',dataset_embeddings.shape)
print('class_activation_mappings : ',class_activation_mappings.shape)
print('129: ',class_activation_mappings[129,:,:])
x = dataset_embeddings[:,0]
y = dataset_embeddings[:,1]
labels = labels.numpy()


convs = model.convs
print(model.conv1.weight.shape)
print(convs[0].weight.shape)



fig, ax = plt.subplots(figsize=(12,3))
plt.title('32 hiddens Class Activation Mapping (CAM) on feature 1 on MUTAG dataset model=NestedGCN')
plt.ylabel('Feature 1 elements')
plt.xlabel('Graph on dataset')

# Plot the first channel
ax.imshow(torch.transpose(class_activation_mappings[:, :, 0],0,1), cmap='gray')

# Add a colorbar for the first channel
cbar = ax.figure.colorbar(ax.imshow(torch.transpose(class_activation_mappings[:, :, 0],0,1), cmap='gray'), ax=ax)

# Show the plot
plt.show()


# print('model.parameters: ',model.parameters())
# print('model.parameters: ',len(list(model.parameters())))  #15
# print('model.parameters: ',(list(model.parameters())[0]).data)  #15 
# tensor([[ 0.4122],
#         [-0.6590],
#         [ 0.1592],
#         [ 0.6859],
#         [-0.0990],
#         [ 0.6898],
#         [ 0.1244],
#         [ 0.6615]])     #torch.Size([8, 1])


for i in range (0,len(list(model.parameters()))):
    print('i: ,',i,' ',((list(model.parameters())[i]).data).shape)


# for i in range (0,len(list(model.parameters()))):
#     print('model.parameters i: ,',i,' ',((list(model.parameters())[i]).data))

# plt.xlabel('Featue 1')    
# plt.ylabel('Featue 2') 
# plt.title('hiddens=32 classification for MUTAG TU dataset using NestedGCN')
# scatter=plt.scatter(x, y, alpha=0.5,c=labels)
# plt.legend(handles=scatter.legend_elements()[0],labels=['-1','1'])

# plt.show()

# def build_adjacency_matrix(edge_index, num_nodes, is_directed=True):
#     if not is_directed:
#         edge_index = to_undirected(edge_index, num_nodes)

#     adj_matrix = torch.zeros((num_nodes, num_nodes))
#     adj_matrix[edge_index[0], edge_index[1]] = 1

#     return adj_matrix

# import networkx as nx
# import matplotlib.pyplot as plt

# # Assuming you have an adjacency matrix or edge list
# adjacency_matrix = ...
# edge_list = ...

# # Create a NetworkX graph from the adjacency matrix or edge list
# graph = nx.from_numpy_matrix(adjacency_matrix)  # or nx.from_edgelist(edge_list)

# # Draw the graph using matplotlib
# nx.draw(graph, with_labels=True)
# plt.show()
