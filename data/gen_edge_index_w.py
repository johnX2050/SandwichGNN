import torch
import scipy.sparse as sp
import numpy as np
from util import *

def ChangeAdjMatrixToEdgeIndex(adj):
    # adj: [N,N] where N = nodes in the graph
    edge_index = torch.nonzero(adj).T
    edge_w = adj[edge_index[0], edge_index[1]]

    return edge_index, edge_w

# get the edge_index and edge_w
adj_data_file = 'adj_mx.pkl'
city_num = 207
predefined_A = load_adj(adj_data_file)
predefined_A = torch.tensor(predefined_A)-torch.eye(city_num)

edge_index, edge_w = ChangeAdjMatrixToEdgeIndex(predefined_A)
np.savez('edge_index.npz', edge_index)
np.savez('edge_w.npz', edge_w)