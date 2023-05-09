import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn.parameter import Parameter
from einops import rearrange, repeat, reduce
from SandwichGNN.attention import FullAttention, AttentionLayer, PositionwiseFeedForward
from SandwichGNN.t_components import RegularMask, Bottleneck_Construct, refer_points, \
    get_mask, PositionwiseFeedForward, MLP
from SandwichGNN.s_components import GNN
from torch.nn.utils import weight_norm
from torch_geometric.nn import GCNConv, norm
from math import ceil


class temporal_feature_modeling_layer(nn.Module):
    """
    Description: This layer is used to extract meaningful temporal representations.
    Input: A two-layer multiscale structure x (B, L + 1/2L, N, D)
    Output: Updated and chosen x (B, 1/2L, N, D)
    """

    def __init__(self, d_model, d_inner, n_heads, dropout=0.1, normalize_before=None):
        super(temporal_feature_modeling_layer, self).__init__()

        self.self_attn = AttentionLayer(
            FullAttention(mask_flag=False, factor=0,
                          attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, n_nodes, input):
        output, _ = self.self_attn(
            input, input, input, attn_mask=None
        )
        output = self.pos_ffn(output)
        output = rearrange(output, '(b n) l d -> b n l d', n=n_nodes)
        return output


class spatial_feature_modeling_layer(nn.Module):
    """
    Description: This layer is used to extract meaningful spatial representations.
                 Using message passing mechanism
    Input: x (B, 1/2L, N, D), edge_index(2, num_edges), edge_w(num_edges, d_edge)
    Output: Updated: x (B, 1/2L, N, D), edge_index(2, num_edges), edge_w(num_edges, d_edge)
    """

    def __init__(self,in_len, d_model):
        super(spatial_feature_modeling_layer, self).__init__()
        self.in_len = in_len
        self.gnn1_embed = GNN(d_model, d_model, d_model)
        self.mlp1 = MLP(d_model, d_model*in_len)
        self.mlp2 = MLP(d_model*in_len, d_model)
        self.d_model = d_model

    def forward(self, x, adj):
        x = rearrange(x, 'b n l d -> b n (l d)')
        x = self.mlp2(x)
        output = self.gnn1_embed(x, adj)
        output = self.mlp1(output)
        return output


class DecoderBlock(nn.Module):
    """
    Description: Compose with four layers
    Input: x (B, L, N, D), edge_index(2, num_edges), edge_w(num_edges, d_edge)
    Output: next_x (B, 1/2L, N/c, D), next_edge_index(2, next_num_edges), next_edge_w(next_num_edges, d_edge)
    """

    def __init__(self, n_blocks, idx_block, n_nodes, window_size, inner_size,
                d_model, d_ff, n_heads, all_len, seq_len=96, dropout=0.2):
        super(DecoderBlock, self).__init__()
        self.seq_len = seq_len
        self.n_nodes = n_nodes
        self.window_size = window_size
        self.inner_size = inner_size
        self.n_blocks = n_blocks
        self.idx_block = idx_block
        self.all_len = all_len
        self.in_len_gnn = 0

        for j in range(self.n_blocks-1, self.idx_block-2, -1):
            self.in_len_gnn += self.all_len[self.n_blocks-j-1]

        self.t_feature = \
            temporal_feature_modeling_layer(d_model=d_model, d_inner=d_ff, n_heads=n_heads, normalize_before=False)

        self.s_feature = \
            spatial_feature_modeling_layer(in_len=self.in_len_gnn, d_model=d_model)



    def forward(self, x, adj):
        n_nodes = x.shape[1]

        x = rearrange(x, 'b n l d -> (b n) l d')
        x_after_t = self.t_feature(n_nodes, x)
        x_after_s = self.s_feature(x_after_t, adj)

        return x_after_s


class Decoder(nn.Module):
    """
    Description: The Encoder compose of four layers.
    Input: x, edge_index, edge_w
    Output: coarsen_x, coarsen_graph(smaller edge_index and edge_w)
    """

    def __init__(self, n_nodes, n_blocks, seq_len, d_x, d_model,d_ff,
                 window_size, inner_size, pred_len,
                 n_heads, d_edge, dropout=0.2
                 ):
        super(Decoder, self).__init__()

        self.n_nodes = n_nodes
        self.n_blocks = n_blocks
        self.seq_len = seq_len
        self.d_x = d_x
        self.d_model = d_model
        self.d_ff = d_ff
        self.window_size = window_size
        self.inner_size = inner_size
        self.pred_len = pred_len
        self.n_heads = n_heads
        self.d_edge = d_edge
        self.dropout = dropout
        self.all_len = []
        self.all_len_sum = 0

        for i in range(n_blocks, 0, -1):
            cur_len = int(self.seq_len // math.pow(2, i))
            self.all_len.append(cur_len)
            self.all_len_sum += cur_len

        self.decode_blocks = nn.ModuleList([
            DecoderBlock(n_blocks=n_blocks, idx_block=n_blocks, n_nodes=self.n_nodes, seq_len=self.seq_len,
                         d_model=self.d_model, d_ff=self.d_ff, window_size=self.window_size,
                         inner_size=self.inner_size, all_len=self.all_len,
                         n_heads=self.n_heads, dropout=self.dropout)]
        )

        for i in range(1, n_blocks):
            self.decode_blocks.append(
                DecoderBlock(n_blocks=n_blocks, idx_block=n_blocks-i, n_nodes=self.n_nodes, seq_len=self.seq_len,
                             d_model=self.d_model, d_ff=self.d_ff, window_size=self.window_size,
                             inner_size=self.inner_size, all_len=self.all_len,
                             n_heads=self.n_heads, dropout=self.dropout)
            )

        self.pred_mlp = Seq(Lin(d_model*self.all_len_sum, 16), ReLU(inplace=True), Lin(16, self.pred_len), ReLU(inplace=True))

    def forward(self, inputs):
        dec_layer_outputs = []
        dec_output = None

        # 0 the decoder block
        x = inputs[0]
        x.reverse()
        adj = inputs[1]
        adj.reverse()
        s = inputs[2]
        s.reverse()

        # for i in range(0, self.n_blocks):
        #     s[i] = rearrange(s[i], 'b small big -> b big small')

        # for 0th decoder_block
        in_len = x[0].shape[2]
        d_model = x[0].shape[3]
        dec_layer_outputs.append(self.decode_blocks[0](x[0], adj[0]))

        dec_next_input = torch.bmm(s[0], dec_layer_outputs[0])
        dec_next_input = rearrange(dec_next_input, 'b n (l d) -> b n l d', l=in_len, d=d_model)

        for i in range(1, self.n_blocks):
            dec_input = torch.cat([x[i], dec_next_input], dim=2)
            in_len = dec_input.shape[2]
            d_model = dec_input.shape[3]
            dec_layer_outputs.append(self.decode_blocks[i](dec_input, adj[i]))

            if i != self.n_blocks-1:
                dec_next_input = torch.bmm(s[i], dec_layer_outputs[i])
                dec_next_input = rearrange(dec_next_input, 'b n (l d) -> b n l d', l=in_len, d=d_model)
            else:
                dec_output = torch.bmm(s[i], dec_layer_outputs[i])

        pred_y = self.pred_mlp(dec_output)

        return pred_y