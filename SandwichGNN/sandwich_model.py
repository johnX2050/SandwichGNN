import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from SandwichGNN.sandwich_encoder import Encoder
from SandwichGNN.sandwich_decoder import Decoder
from cross_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from cross_models.cross_embed import DSW_embedding

from math import ceil


class SandwichGNN(nn.Module):
    def __init__(self, d_x, d_edge, d_model=128, d_ff=256,
                 seq_len=96, pred_len=12, window_size=[2],
                 inner_size=5, s_factor = 4,
                 n_nodes=207, n_heads=4, n_blocks=2,
                 dropout=0.0,device=torch.device('cuda:0')):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super(SandwichGNN, self).__init__()
        self.d_x = d_x
        self.d_edge = d_edge
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.s_factor = s_factor
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.device = device
        self.window_size = window_size
        self.inner_size = inner_size

        # The padding operation to handle invisible sgemnet length
        # self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        # self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        # self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.x_embed = nn.Linear(2, d_x)
        # self.edge_w_embed = nn.Linear(1, self.d_edge)

        # Encoder
        self.encoder = Encoder(n_nodes=self.n_nodes, n_blocks=self.n_blocks, seq_len=self.seq_len, d_x=self.d_x,
                               d_model=self.d_model, d_ff=self.d_ff, d_edge=self.d_edge,
                               window_size=self.window_size, inner_size=self.inner_size,
                               s_factor=s_factor, n_heads=self.n_heads, dropout=dropout)

        # Decoder
        self.decoder = Decoder(n_nodes=self.n_nodes, n_blocks=self.n_blocks, seq_len=self.seq_len, d_x=self.d_x,
                               d_model=self.d_model, d_ff=self.d_ff, d_edge=self.d_edge, pred_len=self.pred_len,
                               window_size=self.window_size, inner_size=self.inner_size,
                               n_heads=self.n_heads, dropout=dropout)

    def forward(self, x, adj, loc):

        x_embed = self.x_embed(x)
        enc_out = self.encoder(x_embed, adj)
        predict_y = self.decoder(enc_out)

        return predict_y