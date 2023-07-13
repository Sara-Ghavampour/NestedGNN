
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
# data = dataset[0].to(device)
loss =0
# with torch.no_grad():
#     out = model(data)
# loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
# print('loss: ',loss)
dataset_embeddings=torch.empty((0,2), dtype=torch.float32)
labels=torch.empty((0), dtype=torch.int32)
for data in test_loader:
        data = data.to(device)
        # print((data.y.view(-1)),type(data.y.view(-1)))
        labels=torch.cat((labels,data.y.view(-1)),axis=0)
        with torch.no_grad():
            out,features = model(data,return_features=True)
            print('features: ',features.shape)
            dataset_embeddings=torch.cat((dataset_embeddings,features),axis=0)
        loss += F.nll_loss(features, data.y.view(-1), reduction='sum').item()
# dataset_embeddings = np.asarray(dataset_embeddings)
print('loss: ',loss / len(test_loader.dataset))
print('dataset_embeddings : ',dataset_embeddings.shape)

print('1 data: ',dataset_embeddings[10,:])
x = dataset_embeddings[:,0]
y = dataset_embeddings[:,1]
labels = labels.numpy()

plt.scatter(x, y, alpha=0.5,c=labels)
plt.show()
# loss / len(loader.dataset)
# model = NestedGCN(dataset, 4, 32, 'spd', False)
# model.load_state_dict(torch.load('./models/nestedgcn2.pt'))

# print(model.parameters())
# print(len(list(model.parameters())))  #15
# print((list(model.parameters())[0]).data)  #15 
# tensor([[ 0.4122],
#         [-0.6590],
#         [ 0.1592],
#         [ 0.6859],
#         [-0.0990],
#         [ 0.6898],
#         [ 0.1244],
#         [ 0.6615]])     #torch.Size([8, 1])

# print(((list(model.parameters())[2]).data).shape)  #15 

# for i in range (0,len(list(model.parameters()))):
#     print('i: ,',i,' ',((list(model.parameters())[i]).data).shape)


# for i in range (0,len(list(model.parameters()))):
#     print('i: ,',i,' ',((list(model.parameters())[i]).data))

def build_adjacency_matrix(edge_index, num_nodes, is_directed=True):
    if not is_directed:
        edge_index = to_undirected(edge_index, num_nodes)

    adj_matrix = torch.zeros((num_nodes, num_nodes))
    adj_matrix[edge_index[0], edge_index[1]] = 1

    return adj_matrix
