import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool
from einops import repeat, reduce, rearrange

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))


    def forward(self, x, adj, mask=None):
        b, n, d = x.size()

        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = rearrange(x, 'b n d -> b d n')
            x = self.bns[step](x)
            x = rearrange(x, 'b d n -> b n d')

        return x