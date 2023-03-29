import math
import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MergeLayer(nn.Module):
    def __init__(self, dim1, dim2, dim4):
        super().__init__()
        self.dim3 = int((dim1 + dim2) / 2)
        self.fc1 = nn.Linear(dim1 + dim2, self.dim3)
        self.fc2 = nn.Linear(self.dim3, dim4)
        self.act = nn.PReLU()
        self.bn = nn.BatchNorm1d(dim4)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        h = self.bn(self.act(self.fc2(h)))
        return h


class TemporalAttentionLayer(torch.nn.Module):
    """
  Temporal attention layer. Return the temporal embedding of a node given the node itself,
   its neighbors and the edge timestamps.
  """

    def __init__(self, n_node_embedding, n_node_old_embedding, n_hidden_embedding,
                 output_dimension, n_head=2,
                 dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()

        self.n_head = n_head

        self.feat_dim = n_node_embedding
        self.embed_dim = n_node_old_embedding
        self.hid_dim = n_hidden_embedding

        self.query_dim = self.feat_dim
        self.key_dim = self.hid_dim

        self.merger = MergeLayer(self.embed_dim, self.feat_dim, output_dimension)

        self.multi_head_target = torch.nn.MultiheadAttention(embed_dim=self.query_dim,
                                                             kdim=self.key_dim,
                                                             vdim=self.key_dim,
                                                             num_heads=n_head,
                                                             dropout=dropout)

    def forward(self, old_node_embedding, node_embedding, hidden_embedding, mask):
        """
    "Temporal attention model
    :param src_node_features: float Tensor of shape [batch_size, n_node_features]
    :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
    :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
    :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors,
    time_dim]
    :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
    :return:
    attn_output: float Tensor of shape [1, batch_size, n_node_features]
    attn_output_weights: [batch_size, 1, n_neighbors]
    """

        query = hidden_embedding
        key = node_embedding
        #print(mask.shape)
        # print(neighbors_features.shape, edge_features.shape, neighbors_time_features.shape)
        # Reshape tensors so to expected shape by multi head attention
        # query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
        # key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]

        # Compute mask of which source nodes have no valid neighbors
        invalid_mask = mask.all(dim=1)
        # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
        # force the attention to just 'attend' on this neighbor (which has the same features as all
        # the others since they are fake neighbors) and will produce an equivalent result to the
        # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
        #mask[invalid_mask.squeeze(), 0] = False

        #print(query.shape, key.shape, invalid_mask.shape)

        attn_output, _ = self.multi_head_target(query=hidden_embedding, key=node_embedding, value=node_embedding,
                                                key_padding_mask=mask.all(dim=1))

        attn_output = attn_output.squeeze()

        # Source nodes with no neighbors have an all zero attention output. The attention output is
        # then added or concatenated to the original source node features and then fed into an MLP.
        # This means that an all zero vector is not used.
        attn_output = attn_output.masked_fill(mask, 0)

        # Skip connection with temporal attention over neighborhood and the features of the node itself
        torch.cuda.empty_cache()
        attn_output = self.merger(attn_output, old_node_embedding)

        return attn_output
