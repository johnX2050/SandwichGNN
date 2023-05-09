import math
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn.parameter import Parameter
from einops import rearrange, repeat, reduce
from SandwichGNN.attention import FullAttention, AttentionLayer, PositionwiseFeedForward
from SandwichGNN.t_components import RegularMask, Bottleneck_Construct, refer_points, \
    get_mask, PositionwiseFeedForward, MLP
from SandwichGNN.s_components import GNN
from torch_geometric.nn import DenseGCNConv, dense_diff_pool



class temporal_multi_scale_construct_layer(nn.Module):
    """
    Description: This layer is used to construct temporal multiscale structure.
    Input: A finer scale x(B, N, L, D)
    Output: A two-layer multiscale structure x(B, N, L + 1/2L, D)
    """

    def __init__(self, d_x_in, window_size, dropout=0.1):
        super(temporal_multi_scale_construct_layer, self).__init__()
        self.d_bottleneck = d_x_in // 4
        self.conv_layers = Bottleneck_Construct(
            d_x_in, window_size, self.d_bottleneck)

    def forward(self, x_embed):
        x_embed = rearrange(x_embed, 'b n l d -> (b n) l d')
        output = self.conv_layers(x_embed)
        return output


class temporal_feature_modeling_layer(nn.Module):
    """
    Description: This layer is used to extract meaningful temporal representations.
    Input: A two-layer multiscale structure x (B, L + 1/2L, N, D)
    Output: Updated and chosen x (B, 1/2L, N, D)
    """

    def __init__(self, d_model, d_inner, n_heads, dropout=0.1, normalize_before=None):
        super(temporal_feature_modeling_layer, self).__init__()

        self.self_attn = AttentionLayer(
            FullAttention(mask_flag=True, factor=0,
                          attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, n_nodes, input, slf_attn_mask=None):
        attn_mask = RegularMask(slf_attn_mask)
        output, _ = self.self_attn(
            input, input, input, attn_mask=attn_mask
        )
        len_input = input.shape[1]
        end_idx = len_input // 3
        output = self.pos_ffn(output)
        output = output[:, -end_idx:, :]
        output = rearrange(output, '(b n) l d -> b n l d', n=n_nodes)
        return output


class spatial_feature_modeling_layer(nn.Module):
    """
    Description: This layer is used to extract meaningful spatial representations.
                 Using message passing mechanism
    Input: x (B, 1/2L, N, D), edge_index(2, num_edges), edge_w(num_edges, d_edge)
    Output: Updated: x (B, 1/2L, N, D), edge_index(2, num_edges), edge_w(num_edges, d_edge)
    """

    def __init__(self, in_len, d_model):
        super(spatial_feature_modeling_layer, self).__init__()
        self.in_len = in_len
        self.gnn1_embed = GNN(d_model, d_model, d_model)
        self.mlp2 = MLP(d_model*in_len, d_model)
        self.d_model = d_model

    def forward(self, x, adj):
        x = rearrange(x, 'b n l d -> b n (l d)')
        x = self.mlp2(x)
        output = self.gnn1_embed(x, adj)
        # output = self.mlp1(output)
        return output

class spatial_multi_scale_construct_layer(nn.Module):
    """
    Description: This layer is used to construct spatial multiscale structure.
    Input: x (B, 1/2L, N, D), edge_index(2, num_edges), edge_w(num_edges, d_edge)
    Output: next_x (B, 1/2L, N/c, D), next_edge_index(2, next_num_edges), next_edge_w(next_num_edges, d_edge)
    """

    def __init__(self, in_len, d_model, d_edge, n_nodes, next_n_nodes):
        super(spatial_multi_scale_construct_layer, self).__init__()
        self.in_len = in_len
        self.d_model = d_model
        # self.edge_inf = Seq(Lin(d_model*2, d_edge), ReLU(inplace=True))
        self.n_nodes = n_nodes
        self.next_n_nodes = next_n_nodes
        # self.s = Parameter(torch.randn(n_nodes, next_n_nodes).
        #                             to('cuda', non_blocking=True), requires_grad=True)
        self.gnn1_pool = GNN(d_model, d_model, next_n_nodes)
        self.mlp1 = MLP(d_model, d_model*in_len)

    def forward(self, x, adj):

        # Diffpool get from pygeometric
        s = self.gnn1_pool(x, adj)
        next_x, next_adj, _, _ = dense_diff_pool(x, adj, s)

        next_x = self.mlp1(next_x)
        next_x = rearrange(next_x, 'b n (d d1) -> b n d d1', d1=self.d_model)

        return next_x, next_adj, s


class EncoderBlock(nn.Module):
    """
    Description: Compose with four layers
    Input: x (B, L, N, D), edge_index(2, num_edges), edge_w(num_edges, d_edge)
    Output: next_x (B, 1/2L, N/c, D), next_edge_index(2, next_num_edges), next_edge_w(next_num_edges, d_edge)
    """

    def __init__(self, idx_block, n_nodes, next_n_nodes, window_size, inner_size,
                 d_x, d_model, d_ff, n_heads, d_edge, seq_len=96, dropout=0.2):
        super(EncoderBlock, self).__init__()
        self.seq_len = seq_len
        self.n_nodes = n_nodes
        self.next_n_nodes = next_n_nodes
        self.window_size = window_size
        self.inner_size = inner_size
        self.idx_block = idx_block
        self.in_len = int(self.seq_len // math.pow(2, self.idx_block))
        self.in_len_gnn = int(self.seq_len // math.pow(2, self.idx_block+1))

        self.t_construct = \
            temporal_multi_scale_construct_layer(d_x_in=d_x, window_size=window_size)
        self.t_feature = \
            temporal_feature_modeling_layer(d_model=d_model, d_inner=d_ff, n_heads=n_heads, normalize_before=False)
        self.s_feature = \
            spatial_feature_modeling_layer(in_len=self.in_len_gnn, d_model=d_model)
        self.s_construct = \
            spatial_multi_scale_construct_layer(in_len=self.in_len_gnn,
                                                d_model=d_model, d_edge=d_edge, n_nodes=n_nodes,
                                                next_n_nodes=next_n_nodes )


    def forward(self, x, adj):
        b_n = x.shape[0] * x.shape[1]

        # get attention mask
        mask, all_size = get_mask(
            self.in_len, self.window_size, self.inner_size)
        self.indexes = refer_points(all_size, self.window_size)
        mask = mask.repeat(b_n, 1, 1).to(x.device)

        n_nodes = x.shape[1]

        x_t = self.t_construct(x)
        x_after_t = self.t_feature(n_nodes, x_t, mask)
        x_after_s = self.s_feature(x_after_t, adj)
        next_x, next_adj, s = self.s_construct(x_after_s, adj)

        return next_x, next_adj, s


class Encoder(nn.Module):
    """
    Description: The Encoder compose of four layers.
    Input: x, edge_index, edge_w
    Output: coarsen_x, coarsen_graph(smaller edge_index and edge_w)
    """

    def __init__(self, n_nodes, n_blocks, seq_len, d_x, d_model,d_ff,
                 window_size, inner_size, s_factor,
                 n_heads, d_edge, dropout=0.2
                 ):
        super(Encoder, self).__init__()

        self.n_nodes = n_nodes
        self.n_blocks = n_blocks
        self.seq_len = seq_len
        self.d_x = d_x
        self.d_model = d_model
        self.d_ff = d_ff
        self.window_size = window_size
        self.inner_size = inner_size
        self.s_factor = s_factor
        self.n_heads = n_heads
        self.d_edge = d_edge
        self.dropout = dropout

        self.encode_blocks = nn.ModuleList([
            EncoderBlock(idx_block=0, n_nodes=self.n_nodes, seq_len=self.seq_len, d_x=self.d_x,
                         d_model=self.d_model, d_ff=self.d_ff, window_size=self.window_size,
                         inner_size=self.inner_size, next_n_nodes= int(n_nodes // s_factor),
                         n_heads=self.n_heads, d_edge=self.d_edge, dropout=self.dropout)]
        )

        for i in range(1, n_blocks):
            cur_n_nodes = int(n_nodes // math.pow(s_factor, i))
            next_n_nodes = int(cur_n_nodes // s_factor)
            self.encode_blocks.append(
                EncoderBlock(idx_block=i, n_nodes=cur_n_nodes, seq_len=self.seq_len, d_x=self.d_x,
                             d_model=self.d_model, d_ff=self.d_ff, window_size=self.window_size,
                             inner_size=self.inner_size, next_n_nodes=next_n_nodes,
                             n_heads=self.n_heads, d_edge=self.d_edge, dropout=self.dropout)
            )

    def forward(self, x, adj):
        enc_x = []
        enc_adj = []
        enc_s = []
        enc_outputs = []

        x, next_adj, s = self.encode_blocks[0](x, adj)
        enc_x.append(x)
        enc_adj.append(next_adj)
        enc_s.append(s)

        for i in range(1, len(self.encode_blocks)):
            x, next_adj, s = self.encode_blocks[i](x, next_adj)
            enc_x.append(x)
            enc_adj.append(next_adj)
            enc_s.append(s)

        enc_outputs.append(enc_x)
        enc_outputs.append(enc_adj)
        enc_outputs.append(enc_s)

        return enc_outputs