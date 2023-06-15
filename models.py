import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.linear = nn.Linear(nclass, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class MLP(nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = nn.Linear(dim, 64)
       # self.fc_2 = nn.Linear(64, 10)
        self.fc_3 = nn.Linear(64, 1)
        nn.init.xavier_normal_(self.fc_1.weight, gain=1.0)
        #nn.init.xavier_normal_(self.fc_2.weight, gain=1.0)
        nn.init.xavier_normal_(self.fc_3.weight, gain=1.0)
        self.bn_1 = nn.BatchNorm1d(64)
        #self.bn_2 = nn.BatchNorm1d(10)
        self.bn_3 = nn.BatchNorm1d(1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.bn_1(self.act(self.fc_1(x)))
        x = self.dropout(x)
        #x = self.bn_2(self.act(self.fc_2(x)))
        #x = self.dropout(x)
        x = self.fc_3(x)
        return x

class GSL(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        params = torch.ones([self.dim, self.dim])
        nn.init.normal_(params)
        self.weight = nn.Parameter(params, requires_grad=True)


    def forward(self, feature):
        n_nodes = feature.shape[0]
        #print(n_nodes, feature.shape, self.weight.shape)
        x = torch.zeros(n_nodes, n_nodes)
        #print(x.shape)
        f = torch.unsqueeze(feature, 1)
        f = f.expand(f.shape[0], f.shape[0], f.shape[2])
        ft = torch.transpose(f, 0, 1)
        x = (f - ft).matmul(self.weight)
        x = torch.sqrt((x * x).sum(2))

        sigma = torch.std(x)
        x = torch.exp(-x / 2*sigma.pow(2))
        adj = nn.functional.normalize(x.view(1, -1), p=2.0, dim=1, eps=1e-12, out=None).view(n_nodes, n_nodes)
        return adj



class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(LSTMNet, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )

    def forward(self, x, h, c):
        h_output, c_output = self.rnn(x, (h, c))  # None 表示 hidden state 会用全0的 state
        h_output = self.dropout(h_output)
        c_output = self.dropout(c_output)
        # print(out.shape)
        return h_output, c_output


class Linear(nn.Module):
    def __init__(self, inputSize, outputSize, dropout):
        super(Linear, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)
        self.batch_norm = nn.BatchNorm1d(outputSize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        # x = self.batch_norm(x)
        out = self.dropout(x)
        return out


