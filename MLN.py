import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from models import GCN, LSTMNet, Linear, MLP, GSL, MergeLayer
from layers import TemporalAttentionLayer


class MLN(torch.nn.Module):
    def __init__(self, datelist, node_feature, adj_viewer, adj_period, adj_description,
                 labels, nodes, nodelist, hidden_dim, device, dropout=0.2, perf=False, gsl=False, alpha=1.0):
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
        self.feature_dim = 11  #10 or 17   11  19
        self.n_nodes = len(self.nodelist)
        self.device = device
        self.perf = perf
        self.gsl = gsl
        self.alpha = alpha

        #self.node_embedding_dict = dict()
        #self.y_pred_dict = dict()
        #self.y_true_dict = dict()

        self.last_hidden_v = np.ones([self.n_nodes, self.hidden], dtype=np.float32)
        self.last_hidden_p = np.ones([self.n_nodes, self.hidden], dtype=np.float32)
        self.last_c_v = np.zeros([self.n_nodes, self.hidden], dtype=np.float32)
        self.last_c_p = np.zeros([self.n_nodes, self.hidden], dtype=np.float32)
        self.hidden_embedding = np.zeros([self.n_nodes, self.hidden], dtype=np.float32)

        self.model_v = GCN(self.hidden, self.hidden * 2, self.hidden, dropout=self.dropout)
        self.model_p = GCN(self.hidden, self.hidden * 2, self.hidden, dropout=self.dropout)
        self.rnn_v = LSTMNet(self.hidden, self.hidden, dropout=self.dropout)
        self.rnn_p = LSTMNet(self.hidden, self.hidden, dropout=self.dropout)
        self.merge = MergeLayer(self.hidden, self.hidden, self.hidden, self.hidden)
        self.linear = Linear(self.feature_dim, self.hidden, dropout=self.dropout)
        self.MLP = MLP(self.hidden, self.dropout)
        if self.gsl:
            self.GSL = GSL(self.hidden)
        self.attention_model = TemporalAttentionLayer(
            n_node_embedding=self.hidden,
            n_node_old_embedding=self.hidden,
            n_hidden_embedding=self.hidden,
            output_dimension=self.hidden,
            n_head=2,
            dropout=self.dropout)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def normalize_tensor(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = torch.sum(mx, 1).clone().detach()
        r_inv = torch.flatten(torch.pow(rowsum, -1))
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
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

        #print(features.shape, features[:3])

        # build symmetric adjacency matrix
        adj_v = adj_v + adj_v.T.multiply(adj_v.T > adj_v) - adj_v.multiply(adj_v.T > adj_v)
        #print(adj_v)
        adj_v = self.normalize(adj_v + sp.eye(adj_v.shape[0]))
        #print(adj_v)

        adj_p = adj_p + adj_p.T.multiply(adj_p.T > adj_p) - adj_p.multiply(adj_p.T > adj_p)
        adj_p = self.normalize(adj_p + sp.eye(adj_p.shape[0]))

        adj_d = self.adj_description  # + np.diag(len(self.nodelist))

        features = self.normalize(features)
        #target = self.labels.query('date == @date & channelId in @channels').copy()
        #sort_map = pd.DataFrame({'channelId': channels, }).reset_index().set_index('channelId')

        #target['map'] = target['channelId'].map(sort_map['index'])
        #target.sort_values('map')

        features = self.sparse_mx_to_torch_sparse_tensor(features).to(self.device)

        adj_v = self.sparse_mx_to_torch_sparse_tensor(adj_v).to(self.device)
        adj_p = self.sparse_mx_to_torch_sparse_tensor(adj_p).to(self.device)
        adj_d = torch.FloatTensor(adj_d).to(self.device)



        return adj_v, adj_p, adj_d, features, channels

    def get_embedding(self, date):
        adj_v_ori, adj_p_ori, adj_d, features, channels = self.loadData(date)

        if self.gsl:
            adj_re = self.GSL(torch.from_numpy(self.hidden_embedding).to(self.device))
            adj_re = adj_re + adj_re.T.multiply(adj_re.T > adj_re) - adj_re.multiply(adj_re.T > adj_re)
            adj_re = self.normalize_tensor(adj_re + torch.eye(adj_re.shape[0]).to(self.device)).to_sparse()


            adj_v = self.alpha * adj_v_ori + (1 - self.alpha) * adj_re
            adj_p = self.alpha * adj_p_ori + (1 - self.alpha) * adj_re
        else:
            adj_v = adj_v_ori
            adj_p = adj_p_ori

        hidden_embedding_f = self.linear(features)

        z_v = self.model_v(hidden_embedding_f, adj_v)
        z_p = self.model_p(hidden_embedding_f, adj_p)



        hidden_embedding_v, c_v = self.rnn_v(z_v,
                                            torch.from_numpy(self.last_hidden_v).to(self.device),
                                            torch.from_numpy(self.last_c_v).to(self.device))
        hidden_embedding_p, c_p = self.rnn_v(z_p,
                                            torch.from_numpy(self.last_hidden_p).to(self.device),
                                            torch.from_numpy(self.last_c_p).to(self.device))


        mask = hidden_embedding_v == 0

        '''
        # Straight
        node_embedding = torch.from_numpy(self.hidden_embedding).to(self.device) + hidden_embedding_f + \
                         torch.mm(adj_d, hidden_embedding_v) - torch.mm(adj_d, hidden_embedding_p)

        '''
        # Multi-head attention
        #hidden_embedding = torch.mm(adj_d, hidden_embedding_v) - torch.mm(adj_d, hidden_embedding_p)
        hidden_embedding = torch.mm(adj_d, self.merge(hidden_embedding_v, hidden_embedding_p))


        node_embedding = self.attention_model(torch.from_numpy(self.hidden_embedding).to(self.device),
                                              hidden_embedding_f,
                                              hidden_embedding,
                                              mask)




        #y_pred = self.MLP(node_embedding)
        node_idx = [self.nodelist.index(x) for x in channels]

        filtered_node_embedding = node_embedding[node_idx]

        filtered_y_pred = self.MLP(node_embedding[node_idx])
        tmp = self.labels.query('date == @date')
        if not self.perf:
            filtered_y_true = torch.from_numpy(np.array([tmp.query('channelId == @x')['target'].values
                                                 for x in channels])).to(self.device)
        else:
            filtered_y_true = torch.from_numpy(np.array([tmp.query('channelId == @x')['perf'].values
                                                         for x in channels])).to(self.device)

        valid_mask = (filtered_y_true < 2.0).nonzero()[:, 0]

        #print(valid_mask)

        filtered_y_true = torch.index_select(filtered_y_true, 0, valid_mask)
        filtered_y_pred = torch.index_select(filtered_y_pred, 0, valid_mask)

        #self.node_embedding_dict[date] = filtered_node_embedding
        #self.y_pred_dict[date] = filtered_y_pred
        #self.y_true_dict[date] = filtered_y_true

        with torch.no_grad():
            self.hidden_embedding = node_embedding.detach().cpu().numpy()
            self.last_hidden_v = hidden_embedding_v.detach().cpu().numpy()
            self.last_hidden_p = hidden_embedding_p.detach().cpu().numpy()
            self.last_c_v = c_v.detach().cpu().numpy()
            self.last_c_p = c_p.detach().cpu().numpy()
        return filtered_node_embedding, filtered_y_pred.float(), filtered_y_true.float(),\
               (adj_v - adj_v_ori).to_dense(), (adj_p - adj_p_ori).to_dense()
