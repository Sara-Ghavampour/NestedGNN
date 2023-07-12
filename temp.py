
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

for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
print('loss: ',loss / len(test_loader.dataset))
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