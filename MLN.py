import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from models import GCN, LSTMNet, Linear, MLP


class MLN(torch.nn.Module):
    def __init__(self, datelist, node_feature, adj_viewer, adj_period, adj_description,
                 labels, nodes, nodelist, hidden_dim, device, dropout=0.2):
        super(MLN, self).__init__()
        self.datelist = datelist
        self.node_feature = node_feature
        self.adj_viewer = adj_viewer
        self.adj_period = adj_period
        self.adj_description = adj_description
        self.labels = labels
        self.nodes = nodes
        self.nodelist = nodelist
        self.dropout = dropout
        self.hidden = hidden_dim
        self.feature_dim = 18
        self.n_nodes = len(self.nodelist)
        self.device = device

        self.node_embedding_dict = dict()
        self.y_pred_dict = dict()
        self.y_true_dict = dict()

        self.last_hidden_v = np.zeros([self.n_nodes, self.hidden], dtype=np.float32)
        self.last_hidden_p = np.zeros([self.n_nodes, self.hidden], dtype=np.float32)
        self.last_c_v = np.zeros([self.n_nodes, self.hidden], dtype=np.float32)
        self.last_c_p = np.zeros([self.n_nodes, self.hidden], dtype=np.float32)
        self.hidden_embedding = np.zeros([self.n_nodes, self.hidden], dtype=np.float32)

        self.model_v = GCN(self.feature_dim, self.hidden * 2, self.hidden, dropout=self.dropout)
        self.model_p = GCN(self.feature_dim, self.hidden * 2, self.hidden, dropout=self.dropout)
        self.rnn_v = LSTMNet(self.hidden, self.hidden, dropout=self.dropout)
        self.rnn_p = LSTMNet(self.hidden, self.hidden, dropout=self.dropout)
        self.linear = Linear(self.feature_dim, self.hidden, dropout=self.dropout)
        self.MLP = MLP(self.hidden, self.dropout)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def loadData(self, date):
        features = self.node_feature[date]
        adj_v = self.adj_viewer[date]
        adj_p = self.adj_period[date]
        channels = self.nodes[date]

        # build symmetric adjacency matrix
        adj_v = adj_v + adj_v.T.multiply(adj_v.T > adj_v) - adj_v.multiply(adj_v.T > adj_v)
        adj_v = self.normalize(adj_v + sp.eye(adj_v.shape[0]))

        adj_p = adj_p + adj_p.T.multiply(adj_p.T > adj_p) - adj_p.multiply(adj_p.T > adj_p)
        adj_p = self.normalize(adj_p + sp.eye(adj_p.shape[0]))

        adj_d = self.adj_description  # + np.diag(len(self.nodelist))

        features = self.normalize(features)
        target = self.labels.query('date == @date & channelId in @channels').copy()
        sort_map = pd.DataFrame({'channelId': channels, }).reset_index().set_index('channelId')

        target['map'] = target['channelId'].map(sort_map['index'])
        target.sort_values('map')

        y_true = target['target'].to_numpy()

        features = self.sparse_mx_to_torch_sparse_tensor(features).to(self.device)
        y_true = torch.LongTensor(y_true).to(self.device)
        adj_v = self.sparse_mx_to_torch_sparse_tensor(adj_v).to(self.device)
        adj_p = self.sparse_mx_to_torch_sparse_tensor(adj_p).to(self.device)
        adj_d = torch.FloatTensor(adj_d).to(self.device)

        return adj_v, adj_p, adj_d, features, y_true, channels

    def get_embedding(self, date):
        adj_v, adj_p, adj_d, features, y_true, channels = self.loadData(date)

        hidden_embedding_v, c_v = self.rnn_v(self.model_v(features, adj_v),
                                            torch.from_numpy(self.last_hidden_v).to(self.device),
                                            torch.from_numpy(self.last_c_v).to(self.device))
        hidden_embedding_p, c_p = self.rnn_v(self.model_p(features, adj_p),
                                            torch.from_numpy(self.last_hidden_p).to(self.device),
                                            torch.from_numpy(self.last_c_p).to(self.device))
        hidden_embedding_f = self.linear(features)

        node_embedding = torch.from_numpy(self.hidden_embedding).to(self.device) + hidden_embedding_f + \
                         torch.mm(adj_d, hidden_embedding_v) - torch.mm(adj_d, hidden_embedding_p)


        y_pred = self.MLP(node_embedding)
        node_idx = [self.nodelist.index(x) for x in channels]

        filtered_node_embedding = node_embedding[node_idx]
        filtered_y_pred = y_pred[node_idx]
        tmp = self.labels.query('date == @date')
        filtered_y_true = torch.tensor(np.array([tmp.query('channelId == @x')['target'].values
                                                 for x in channels])).to(self.device)

        self.node_embedding_dict[date] = filtered_node_embedding
        self.y_pred_dict[date] = filtered_y_pred
        self.y_true_dict[date] = filtered_y_true

        with torch.no_grad():
            self.hidden_embedding = node_embedding.detach().cpu().numpy()
            self.last_hidden_v = hidden_embedding_v.detach().cpu().numpy()
            self.last_hidden_p = hidden_embedding_p.detach().cpu().numpy()
            self.last_c_v = c_v.detach().cpu().numpy()
            self.last_c_p = c_p.detach().cpu().numpy()
        return filtered_node_embedding, filtered_y_pred.float(), filtered_y_true.float()
