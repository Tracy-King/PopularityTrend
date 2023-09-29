from __future__ import division
import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import os, re
import argparse
from sklearn.preprocessing import scale
from MLN import MLN

import baseline_LR
import baseline_seq
import torch

parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('--network', type=str, default="wikipedia",
                    choices=['mooc', 'reddit', 'wikipedia'], help='Network name')
parser.add_argument('--days', type=int, default=5, help='Days in a graph')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

network = args.network
datapath = '{}.csv'.format(network)

device_string = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)


def load_network(network, datapath, time_scaling=True):
    '''
    This function loads the input network.

    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions.
    '''

    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []

    print("\n\n**** Loading %s network from file: %s ****" % (network, datapath))
    f = open(datapath, "r")
    f.readline()
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]) - start_timestamp)
        y_true_labels.append(int(ls[3]))  # label = 1 at state change, 0 otherwise
        feature_sequence.append(list(map(float, ls[4:])))
    f.close()

    user_sequence = np.array(user_sequence)
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formating item sequence")
    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formating user sequence")
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
            item2id, item_sequence_id, item_timedifference_sequence, \
            timestamp_sequence, \
            feature_sequence, \
            y_true_labels]


def generateGraph(user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
                  item2id, item_sequence_id, item_timediffs_sequence, timestamp_sequence, feature_sequence, y_true,
                  period):
    feature_sequence = np.array(feature_sequence)
    y_true = np.array(y_true)

    feature_dim = feature_sequence.shape[1]
    total_length = timestamp_sequence[-1]
    num_graphs = int(total_length / period)
    period_list = [i*period for i in list(range(num_graphs + 1))]

    barrier_list = []

    for cut_time in period_list:
        i = np.searchsorted(timestamp_sequence, cut_time)
        barrier_list.append(i)
    if barrier_list[-1] < len(timestamp_sequence):
        barrier_list.append(len(timestamp_sequence))
        num_graphs += 1

    num_users = len(user2id)
    num_items = len(item2id)

    #print(barrier_list, period_list)

    adj_dict = {}
    biadj_dict = {}
    label_dict = {}
    item_dict = {}
    item_feature_dict = {}
    user_feature_dict = {}
    empty_adj_dict = {}

    for k in range(num_graphs):
        start_idx = barrier_list[k]
        end_idx = barrier_list[k + 1]

        users = list(set(user_sequence_id[start_idx:end_idx]))
        items = list(set(item_sequence_id[start_idx:end_idx]))

        label = np.zeros(num_items)
        adj = np.zeros([num_items, num_users])
        item_feature = np.zeros([num_items, feature_dim])
        user_feature = np.zeros([num_users, feature_dim])
        for idx in range(start_idx, end_idx):
            user = user_sequence_id[idx]
            item = item_sequence_id[idx]
            adj[item, user] += y_true[idx]
            label[item] = y_true[idx]
            item_feature[item] = feature_sequence[idx]
            user_feature[user] = feature_sequence[idx]

        intra_adj = np.matmul(adj, adj.T)
        empty_adj = np.array(intra_adj.shape)

        adj_dict[k] = intra_adj
        biadj_dict[k] = adj
        label_dict[k] = label
        item_feature_dict[k] = item_feature
        user_feature_dict[k] = user_feature
        item_dict[k] = items
        empty_adj_dict[k] = empty_adj

        print("*** Graph {}/{} completed ***\n\n".format(k, num_graphs))

    datelist = list(range(num_graphs))

    return datelist, item_feature_dict, adj_dict, label_dict, item_dict, user_feature_dict, biadj_dict, empty_adj_dict


def readNetwork(network='mooc', days=args.days):
    datapath = '{}.csv'.format(network)
    period = 3600 * 24 * days  # 5 days
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
     item2id, item_sequence_id, item_timediffs_sequence,
     timestamp_sequence, feature_sequence, y_true] = load_network(network, datapath, time_scaling=True)

    datelist, item_feature_dict, adj_dict, label_dict, item_dict, user_feature_dict, biadj_dict, empty_adj_dict = \
        generateGraph(user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
                      item2id, item_sequence_id, item_timediffs_sequence,
                      timestamp_sequence, feature_sequence, y_true, period)

    return datelist, item_feature_dict, adj_dict, adj_dict, empty_adj_dict[0], label_dict, item_dict, item2id, user_feature_dict, biadj_dict


def LRbaseline():
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
     item2id, item_sequence_id, item_timediffs_sequence,
     timestamp_sequence, feature_sequence, y_true] = load_network(network, datapath, time_scaling=True)

    x = np.array(feature_sequence)
    y = np.array(y_true)

    mean = x.mean(axis=0)
    x -= mean  # 等价于 train_data = train_data - mean
    std = x.std(axis=0)
    x / (1 + std)
    #print('x:', mean, std)

    mean = y.mean(axis=0)
    # y -= mean  # 等价于 train_data = train_data - mean
    std = y.std(axis=0)
    # y / (1 + std)
    #print('y', mean, std)

    t = 0.1
    a = 1
    baseline_LR.LR(x, y, t, mean, std)
    baseline_LR.SGD(x, y, t, mean, std)
    baseline_LR.GBR(x, y, t, mean, std)
    baseline_LR.XGBoost(x, y, t, mean, std)
    baseline_LR.NN(x, y, t, mean, std)


def Seqbaseline():
    datelist, item_feature_dict, adj_dict, adj_dict, empty_adj, label_dict, item_dict, item2id, user_feature_dict, biadj_dict = \
        readNetwork(network)

    x = [item_feature_dict[i].astype('float32') for i in datelist]
    y = [label_dict[i].astype('float32') for i in datelist]

    t = 0.1
    layer = 2

    # LSTMFCN(x, y, test_size=t, num_layers=layer)
    #baseline_seq.LSTM(x, y, test_size=t, num_layers=layer)
    #baseline_seq.GRU(x, y, test_size=t, num_layers=layer)
    #baseline_seq.RNN(x, y, test_size=t, num_layers=layer)
    baseline_seq.LSTM_FCN(x, y, test_size=t, num_layers=layer)


def train():
    [datelist, item_feature_list, adj_list, empty_adj_list, label_list, item2id, item_list, user_feature_list, biadj_list]\
        = readNetwork(network, datapath, args.days)
    model = MLN(datelist, item_feature_list, adj_list, adj_list, empty_adj_list[0],
                label_list, item2id, item_list, user_feature_list, biadj_list, 128, device)
    model = model.to(device)




if __name__ == "__main__":
    #LRbaseline()
    Seqbaseline()
