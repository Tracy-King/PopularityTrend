import pandas as pd
import numpy as np
import glob
import sys
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
#from plotly.subplots import make_subplot
from datetime import datetime
import argparse
import xgboost
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import math
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
from lstmfcn import LSTMFCN
import itertools


pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)


parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('--start', type=str, default="2021-04", help='Start date(e.g. 2021-04)')
parser.add_argument('--period', type=str, default="d", choices=[
  "d", "w", "m"], help='Period of data separation(day, week, month)')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

PERIOD = args.period
START = args.start

f = open('logs_{}_seq.txt'.format(PERIOD), 'w')


'''
df = pd.concat([
    pd.read_parquet(f)
    for f in iglob('../input/vtuber-livechat/superchats_*.parquet')], ignore_index=True)
'''


def dataSplit(df):
    df['target'] = df.groupby(['channelId'])['totalSC'].shift(-1)
    df = df.query('target == target')
    df['perf'] = (df['target'] / df['totalSC']) - 1

    df = df.query('perf <= 2.0')


    dfgroup = df.groupby(['channelId'])
    channelList = df['channelId'].drop_duplicates().tolist()
    #dateList = df['date'].drop_duplicates().tolist()

    x = []
    y = []
    seq_length = []

    for channel in channelList:
        tmp = dfgroup.get_group(channel).sort_values(by=['date'])
        #print(tmp)
        x_ = tmp.drop(['date', 'channelId', 'target', 'Unnamed: 0', 'perf'], axis=1).astype('float32').to_numpy()
        y_ = tmp['perf'].astype('float32').to_numpy()   #[-1]
        length = tmp.shape[0]
        #print(x_, y_, length)
        #print('___________________________')

        x.append(x_)
        y.append(y_)
        #print(x_.shape, y_.shape, length)
        #print('channel {} finished.'.format(channel))

    #df.to_csv('tmp.csv')

    return x, y


def collate_fn(data):
    data.sort(key=lambda x: x.shape[0], reverse=True)
    seq_len = [s.shape[0] for s in data]
    data = pad_sequence(data, batch_first=True)
    #data = pack_padded_sequence(data, seq_len, batch_first=True)
    return data, seq_len

def LSTM_old(x, y, test_size, hidden_size=8, num_layers=3):
    class LSTMNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(LSTMNet, self).__init__()
            self.dropout = torch.nn.Dropout(0.2)
            self.rnn = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.out = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 1)
            )

        def forward(self, x):
            packed_out, _ = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
            r_out, len_seq = pad_packed_sequence(packed_out, batch_first=True)
            len_idx = len_seq - 1
            len_idx = torch.unsqueeze(len_idx, 1).cuda()
            r_out = self.dropout(r_out)
            out = torch.squeeze(self.out(r_out))
            out = torch.gather(out, 1, len_idx)
            #print(out.shape)
            return out, len_seq


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)


    mean = sum([x.mean(axis=0) for x in X_train])/len(X_train)
    X_train = [x - mean for x in X_train]  # 等价于 train_data = train_data - mean
    std = sum([x.std(axis=0) for x in X_train])/len(X_train)
    #print(mean, std)
    X_train = [(x/(1+std)) for x in X_train]

    X_test = [x - mean for x in X_test]  # 训练集的均值和标准差
    X_test = [(x/(1+std)) for x in X_test]



    padded_x_train, seq_len_x_train = collate_fn([torch.from_numpy(x) for x in X_train])
    padded_x_test, seq_len_x_test = collate_fn([torch.from_numpy(x) for x in X_test])


    inputDim = X_train[0].shape[1]  # takes variable 'x'
    learningRate = 0.1
    epochs = 100
    batch_size = 2000
    num_instance = len(X_train)
    num_batch = math.ceil(num_instance / batch_size)

    model = LSTMNet(inputDim, hidden_size, num_layers)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.01)



    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(padded_x_train.cuda())
            labels = Variable(torch.from_numpy(np.array(y_train)).unsqueeze(1).cuda())
        else:
            inputs = Variable(padded_x_train)
            labels = Variable(torch.from_numpy(np.array(y_train)).unsqueeze(1))
        #print(inputs.shape, labels.shape)
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        for k in range(num_batch):
            loss = 0
            s_idx = k*batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            inputs_batch = inputs[s_idx:e_idx]
            input_seq_len_batch = seq_len_x_train[s_idx:e_idx]
            labels_batch = labels[s_idx:e_idx]

            # get output from the model, given the inputs
            outputs = model(pack_padded_sequence(inputs_batch, input_seq_len_batch, batch_first=True))


            # get loss for the predicted output
            loss += criterion(outputs, labels_batch)
            if loss != loss:
                print('loss={}'.format(loss))
                print(torch.isnan(outputs).int().sum())
                print(torch.isnan(labels_batch).int().sum())
                print(s_idx, e_idx, k, num_batch)
                print(outputs, labels_batch)
                print(inputs_batch, labels_batch)
                break
            # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))

    with torch.no_grad():  # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            y_pred = model(pack_padded_sequence(Variable(padded_x_test.cuda()), seq_len_x_test, batch_first=True)).cpu().data.numpy()
        else:
            y_pred = model(pack_padded_sequence(Variable(padded_x_test), seq_len_x_test, batch_first=True)).data.numpy()

        print(y_pred.shape)

        rmse = math.sqrt(mean_squared_error(np.array(y_test), y_pred))
        mape = mean_absolute_percentage_error(np.array(y_test), y_pred)

        print('LSTM----(t={}, l={})------'.format(test_size, num_layers))
        print('RMSE: {}, MAPE: {}'.format(rmse, mape))
        f.write('LSTM---(t={}, l={})--------\n'.format(test_size, num_layers))
        f.write('RMSE: {}, MAPE: {}\n'.format(rmse, mape))


    return


def GRU_old(x, y, test_size, hidden_size=8, num_layers=3):
    class GRUNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(GRUNet, self).__init__()
            self.dropout = torch.nn.Dropout(0.2)
            self.rnn = torch.nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.out = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 1)
            )

        def forward(self, x):
            packed_out, _ = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
            r_out, len_seq = pad_packed_sequence(packed_out, batch_first=True)
            len_idx = len_seq - 1
            len_idx = torch.unsqueeze(len_idx, 1).cuda()
            #print(len_idx, len_idx.shape)
            r_out = self.dropout(r_out)
            out = torch.squeeze(self.out(r_out))
            #print(out.shape)
            out = torch.gather(out, 1, len_idx)
            #print(out.shape)
            return out

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)


    mean = sum([x.mean(axis=0) for x in X_train])/len(X_train)
    X_train = [x - mean for x in X_train]  # 等价于 train_data = train_data - mean
    std = sum([x.std(axis=0) for x in X_train])/len(X_train)
    #print(mean, std)
    X_train = [(x/(1+std)) for x in X_train]

    X_test = [x - mean for x in X_test]  # 训练集的均值和标准差
    X_test = [(x/(1+std)) for x in X_test]



    padded_x_train, seq_len_x_train = collate_fn([torch.from_numpy(x) for x in X_train])
    padded_x_test, seq_len_x_test = collate_fn([torch.from_numpy(x) for x in X_test])


    inputDim = X_train[0].shape[1]  # takes variable 'x'
    learningRate = 0.1
    epochs = 100
    batch_size = 2000
    num_instance = len(X_train)
    num_batch = math.ceil(num_instance / batch_size)

    model = GRUNet(inputDim, hidden_size, num_layers)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.01)



    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(padded_x_train.cuda())
            labels = Variable(torch.from_numpy(np.array(y_train)).unsqueeze(1).cuda())
        else:
            inputs = Variable(padded_x_train)
            labels = Variable(torch.from_numpy(np.array(y_train)).unsqueeze(1))
        #print(inputs.shape, labels.shape)
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        for k in range(num_batch):
            loss = 0
            s_idx = k*batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            inputs_batch = inputs[s_idx:e_idx]
            input_seq_len_batch = seq_len_x_train[s_idx:e_idx]
            labels_batch = labels[s_idx:e_idx]

            # get output from the model, given the inputs
            outputs = model(pack_padded_sequence(inputs_batch, input_seq_len_batch, batch_first=True))


            # get loss for the predicted output
            loss += criterion(outputs, labels_batch)
            if loss != loss:
                print('loss={}'.format(loss))
                print(torch.isnan(outputs).int().sum())
                print(torch.isnan(labels_batch).int().sum())
                print(s_idx, e_idx, k, num_batch)
                print(outputs, labels_batch)
                print(inputs_batch, labels_batch)
                break
            # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))

    with torch.no_grad():  # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            y_pred = model(pack_padded_sequence(Variable(padded_x_test.cuda()), seq_len_x_test, batch_first=True)).cpu().data.numpy()
        else:
            y_pred = model(pack_padded_sequence(Variable(padded_x_test), seq_len_x_test, batch_first=True)).data.numpy()

        print(y_pred.shape)

        rmse = math.sqrt(mean_squared_error(np.array(y_test), y_pred))
        mape = mean_absolute_percentage_error(np.array(y_test), y_pred)

        print('GRU----(t={}, l={})------'.format(test_size, num_layers))
        print('RMSE: {}, MAPE: {}'.format(rmse, mape))
        f.write('GRU---(t={}, l={})--------\n'.format(test_size, num_layers))
        f.write('RMSE: {}, MAPE: {}\n'.format(rmse, mape))


    return



def RNN_old(x, y, test_size, hidden_size=8, num_layers=3):
    class RNNNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(RNNNet, self).__init__()
            self.dropout = torch.nn.Dropout(0.2)
            self.rnn = torch.nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.out = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 1)
            )

        def forward(self, x):
            packed_out, _ = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
            r_out, len_seq = pad_packed_sequence(packed_out, batch_first=True)
            len_idx = len_seq - 1
            len_idx = torch.unsqueeze(len_idx, 1).cuda()
            #print(len_idx, len_idx.shape)
            r_out = self.dropout(r_out)
            out = torch.squeeze(self.out(r_out))
            #print(out.shape)
            out = torch.gather(out, 1, len_idx)
            #print(out.shape)
            return out

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)


    mean = sum([x.mean(axis=0) for x in X_train])/len(X_train)
    X_train = [x - mean for x in X_train]  # 等价于 train_data = train_data - mean
    std = sum([x.std(axis=0) for x in X_train])/len(X_train)
    #print(mean, std)
    X_train = [(x/(1+std)) for x in X_train]

    X_test = [x - mean for x in X_test]  # 训练集的均值和标准差
    X_test = [(x/(1+std)) for x in X_test]



    padded_x_train, seq_len_x_train = collate_fn([torch.from_numpy(x) for x in X_train])
    padded_x_test, seq_len_x_test = collate_fn([torch.from_numpy(x) for x in X_test])


    inputDim = X_train[0].shape[1]  # takes variable 'x'
    learningRate = 0.1
    epochs = 100
    batch_size = 2000
    num_instance = len(X_train)
    num_batch = math.ceil(num_instance / batch_size)

    model = RNNNet(inputDim, hidden_size, num_layers)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.01)



    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(padded_x_train.cuda())
            labels = Variable(torch.from_numpy(np.array(y_train)).unsqueeze(1).cuda())
        else:
            inputs = Variable(padded_x_train)
            labels = Variable(torch.from_numpy(np.array(y_train)).unsqueeze(1))
        #print(inputs.shape, labels.shape)
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        for k in range(num_batch):
            loss = 0
            s_idx = k*batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            inputs_batch = inputs[s_idx:e_idx]
            input_seq_len_batch = seq_len_x_train[s_idx:e_idx]
            labels_batch = labels[s_idx:e_idx]

            # get output from the model, given the inputs
            outputs = model(pack_padded_sequence(inputs_batch, input_seq_len_batch, batch_first=True))


            # get loss for the predicted output
            loss += criterion(outputs, labels_batch)
            if loss != loss:
                print('loss={}'.format(loss))
                print(torch.isnan(outputs).int().sum())
                print(torch.isnan(labels_batch).int().sum())
                print(s_idx, e_idx, k, num_batch)
                print(outputs, labels_batch)
                print(inputs_batch, labels_batch)
                break
            # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))

    with torch.no_grad():  # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            y_pred = model(pack_padded_sequence(Variable(padded_x_test.cuda()), seq_len_x_test, batch_first=True)).cpu().data.numpy()
        else:
            y_pred = model(pack_padded_sequence(Variable(padded_x_test), seq_len_x_test, batch_first=True)).data.numpy()

        print(y_pred.shape)

        rmse = math.sqrt(mean_squared_error(np.array(y_test), y_pred))
        mape = mean_absolute_percentage_error(np.array(y_test), y_pred)

        print('RNN----(t={}, l={})------'.format(test_size, num_layers))
        print('RMSE: {}, MAPE: {}'.format(rmse, mape))
        f.write('RNN---(t={}, l={})--------\n'.format(test_size, num_layers))
        f.write('RMSE: {}, MAPE: {}\n'.format(rmse, mape))


    return



def LSTM(x, y, test_size, hidden_size=8, num_layers=3):
    class LSTMNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(LSTMNet, self).__init__()
            self.dropout = torch.nn.Dropout(0.2)
            self.rnn = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.out = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 1)
            )

        def forward(self, x):
            packed_out, _ = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
            r_out, len_seq = pad_packed_sequence(packed_out, batch_first=True)
            len_idx = len_seq - 1
            len_idx = torch.unsqueeze(len_idx, 1).cuda()
            r_out = self.dropout(r_out)
            out = torch.squeeze(self.out(r_out))
            #out = torch.gather(out, 1, len_idx)
            #print(out.shape)
            return out, len_seq

    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    x = [i for i in x if i.shape[0]>1]
    y = [i for i in y if i.shape[0]>1]


    X_train = [i[:-1] for i in x]
    y_train = [i[:-1] for i in y]
    X_test = x
    y_test = y

    print(X_train[0].shape, y_train[0].shape, X_test[0].shape, y_test[0].shape)



    mean = sum([x.mean(axis=0) for x in X_train])/len(X_train)
    X_train = [x - mean for x in X_train]  # 等价于 train_data = train_data - mean
    std = sum([x.std(axis=0) for x in X_train])/len(X_train)
    #print(mean, std)
    X_train = [(x/(1+std)) for x in X_train]

    X_test = [x - mean for x in X_test]  # 训练集的均值和标准差
    X_test = [(x/(1+std)) for x in X_test]



    padded_x_train, seq_len_x_train = collate_fn([torch.from_numpy(x).float() for x in X_train])
    padded_x_test, seq_len_x_test = collate_fn([torch.from_numpy(x).float() for x in X_test])

    padded_y_train, seq_len_y_train = collate_fn([torch.from_numpy(x).float() for x in y_train])
    padded_y_test, seq_len_y_test = collate_fn([torch.from_numpy(x).float() for x in y_test])


    inputDim = X_train[0].shape[1]  # takes variable 'x'
    learningRate = 0.1
    epochs = 100
    batch_size = 2000
    num_instance = len(X_train)
    num_batch = math.ceil(num_instance / batch_size)

    model = LSTMNet(inputDim, hidden_size, num_layers)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.01)



    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(padded_x_train.cuda())
            #labels = Variable(torch.from_numpy(np.array(padded_y_train)).unsqueeze(1).cuda())
            labels = Variable(padded_y_train.cuda())
        else:
            inputs = Variable(padded_x_train)
            #labels = Variable(torch.from_numpy(np.array(padded_y_train)).unsqueeze(1))
            labels = Variable(padded_y_train)
        #print(inputs.shape, labels.shape)
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        for k in range(num_batch):
            loss = 0
            s_idx = k*batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            inputs_batch = inputs[s_idx:e_idx]
            input_seq_len_batch = seq_len_x_train[s_idx:e_idx]
            labels_batch = labels[s_idx:e_idx]

            # get output from the model, given the inputs
            outputs, out_len_seq = model(pack_padded_sequence(inputs_batch, input_seq_len_batch, batch_first=True))
            #print(outputs.shape, labels_batch.shape)

            # get loss for the predicted output
            loss += criterion(outputs, labels_batch)
            if loss != loss:
                print('loss={}'.format(loss))
                print(torch.isnan(outputs).int().sum())
                print(torch.isnan(labels_batch).int().sum())
                print(s_idx, e_idx, k, num_batch)
                print(outputs, labels_batch)
                print(inputs_batch, labels_batch)
                break
            # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        #print('epoch {}, loss {}'.format(epoch, loss.item()))


    with torch.no_grad():  # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            y_pred, y_pred_out_len_seq = model(pack_padded_sequence(Variable(padded_x_test.cuda()), seq_len_x_test, batch_first=True))
            #print(y_pred.shape, y_pred_out_len_seq.shape)
            y_pred = torch.gather(y_pred, 1, torch.unsqueeze(y_pred_out_len_seq - 1, 1).cuda())
            #y_pred = torch.gather(torch.squeeze(y_pred), 1, torch.unsqueeze(y_pred_out_len_seq - 1, 1).cuda())
            y_pred = y_pred.cpu().data.numpy()
            y_true = torch.gather(padded_y_test.cuda(), 1, torch.unsqueeze(torch.tensor(seq_len_y_test, dtype=torch.int64) - 1, 1).cuda())

        else:
            y_pred, y_pred_out_len_seq = model(pack_padded_sequence(Variable(padded_x_test), seq_len_x_test, batch_first=True))
            y_pred = torch.gather(y_pred, 1, torch.unsqueeze(y_pred_out_len_seq-1, 1))
            y_pred = y_pred.data.numpy()
            y_true = torch.gather(padded_y_test, 1, torch.unsqueeze(torch.tensor(seq_len_y_test, dtype=torch.int64) - 1, 1))

        #y_true = torch.gather(padded_y_test, 1, torch.unsqueeze(torch.Tensor(seq_len_y_test) - 1, 1))
        y_true = y_true.cpu().data.numpy()

        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)

        print('LSTM----(t={}, l={})------'.format(test_size, num_layers))
        print('RMSE: {}, MAPE: {}'.format(rmse, mape))
        f.write('LSTM---(t={}, l={})--------\n'.format(test_size, num_layers))
        f.write('RMSE: {}, MAPE: {}\n'.format(rmse, mape))


    return


def GRU(x, y, test_size, hidden_size=8, num_layers=3):
    class GRUNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(GRUNet, self).__init__()
            self.dropout = torch.nn.Dropout(0.2)
            self.rnn = torch.nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.out = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 1)
            )

        def forward(self, x):
            packed_out, _ = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
            r_out, len_seq = pad_packed_sequence(packed_out, batch_first=True)
            len_idx = len_seq - 1
            len_idx = torch.unsqueeze(len_idx, 1).cuda()
            #print(len_idx, len_idx.shape)
            r_out = self.dropout(r_out)
            out = torch.squeeze(self.out(r_out))
            #print(out.shape)
            #out = torch.gather(out, 1, len_idx)
            #print(out.shape)
            return out, len_seq

    x = [i for i in x if i.shape[0]>1]
    y = [i for i in y if i.shape[0]>1]


    X_train = [i[:-1] for i in x]
    y_train = [i[:-1] for i in y]
    X_test = x
    y_test = y


    mean = sum([x.mean(axis=0) for x in X_train])/len(X_train)
    X_train = [x - mean for x in X_train]  # 等价于 train_data = train_data - mean
    std = sum([x.std(axis=0) for x in X_train])/len(X_train)

    X_train = [(x/(1+std)) for x in X_train]

    X_test = [x - mean for x in X_test]  # 训练集的均值和标准差
    X_test = [(x/(1+std)) for x in X_test]



    padded_x_train, seq_len_x_train = collate_fn([torch.from_numpy(x).float() for x in X_train])
    padded_x_test, seq_len_x_test = collate_fn([torch.from_numpy(x).float() for x in X_test])

    padded_y_train, seq_len_y_train = collate_fn([torch.from_numpy(x).float() for x in y_train])
    padded_y_test, seq_len_y_test = collate_fn([torch.from_numpy(x).float() for x in y_test])


    inputDim = X_train[0].shape[1]  # takes variable 'x'
    learningRate = 0.1
    epochs = 100
    batch_size = 2000
    num_instance = len(X_train)
    num_batch = math.ceil(num_instance / batch_size)

    model = GRUNet(inputDim, hidden_size, num_layers)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.01)



    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(padded_x_train.cuda())
            #labels = Variable(torch.from_numpy(np.array(padded_y_train)).unsqueeze(1).cuda())
            labels = Variable(padded_y_train.cuda())
        else:
            inputs = Variable(padded_x_train)
            #labels = Variable(torch.from_numpy(np.array(padded_y_train)).unsqueeze(1))
            labels = Variable(padded_y_train)
        #print(inputs.shape, labels.shape)
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        for k in range(num_batch):
            loss = 0
            s_idx = k*batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            inputs_batch = inputs[s_idx:e_idx]
            input_seq_len_batch = seq_len_x_train[s_idx:e_idx]
            labels_batch = labels[s_idx:e_idx]

            # get output from the model, given the inputs
            outputs, out_len_seq = model(pack_padded_sequence(inputs_batch, input_seq_len_batch, batch_first=True))


            # get loss for the predicted output
            loss += criterion(outputs, labels_batch)
            if loss != loss:
                print('loss={}'.format(loss))
                print(torch.isnan(outputs).int().sum())
                print(torch.isnan(labels_batch).int().sum())
                print(s_idx, e_idx, k, num_batch)
                print(outputs, labels_batch)
                print(inputs_batch, labels_batch)
                break
            # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        #print('epoch {}, loss {}'.format(epoch, loss.item()))


    with torch.no_grad():  # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            y_pred, y_pred_out_len_seq = model(pack_padded_sequence(Variable(padded_x_test.cuda()), seq_len_x_test, batch_first=True))
            y_pred = torch.gather(y_pred, 1, torch.unsqueeze(y_pred_out_len_seq - 1, 1).cuda())
            y_pred = y_pred.cpu().data.numpy()
            y_true = torch.gather(padded_y_test.cuda(), 1, torch.unsqueeze(torch.tensor(seq_len_y_test, dtype=torch.int64) - 1, 1).cuda())

        else:
            y_pred, y_pred_out_len_seq = model(pack_padded_sequence(Variable(padded_x_test), seq_len_x_test, batch_first=True))
            y_pred = torch.gather(y_pred, 1, torch.unsqueeze(y_pred_out_len_seq-1, 1))
            y_pred = y_pred.data.numpy()
            y_true = torch.gather(padded_y_test, 1, torch.unsqueeze(torch.tensor(seq_len_y_test, dtype=torch.int64) - 1, 1))

        #y_true = torch.gather(padded_y_test, 1, torch.unsqueeze(torch.Tensor(seq_len_y_test) - 1, 1))
        y_true = y_true.cpu().data.numpy()

        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)

        print('GRU----(t={}, l={})------'.format(test_size, num_layers))
        print('RMSE: {}, MAPE: {}'.format(rmse, mape))
        f.write('GRU---(t={}, l={})--------\n'.format(test_size, num_layers))
        f.write('RMSE: {}, MAPE: {}\n'.format(rmse, mape))
    return



def RNN(x, y, test_size, hidden_size=8, num_layers=3):
    class RNNNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(RNNNet, self).__init__()
            self.dropout = torch.nn.Dropout(0.2)
            self.rnn = torch.nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.out = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 1)
            )

        def forward(self, x):
            packed_out, _ = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
            r_out, len_seq = pad_packed_sequence(packed_out, batch_first=True)
            len_idx = len_seq - 1
            len_idx = torch.unsqueeze(len_idx, 1).cuda()
            #print(len_idx, len_idx.shape)
            r_out = self.dropout(r_out)
            out = torch.squeeze(self.out(r_out))
            #print(out.shape)
            #out = torch.gather(out, 1, len_idx)
            #print(out.shape)
            return out, len_seq

    x = [i for i in x if i.shape[0]>1]
    y = [i for i in y if i.shape[0]>1]


    X_train = [i[:-1] for i in x]
    y_train = [i[:-1] for i in y]
    X_test = x
    y_test = y


    mean = sum([x.mean(axis=0) for x in X_train])/len(X_train)
    X_train = [x - mean for x in X_train]  # 等价于 train_data = train_data - mean
    std = sum([x.std(axis=0) for x in X_train])/len(X_train)

    X_train = [(x/(1+std)) for x in X_train]

    X_test = [x - mean for x in X_test]  # 训练集的均值和标准差
    X_test = [(x/(1+std)) for x in X_test]



    padded_x_train, seq_len_x_train = collate_fn([torch.from_numpy(x).float() for x in X_train])
    padded_x_test, seq_len_x_test = collate_fn([torch.from_numpy(x).float() for x in X_test])

    padded_y_train, seq_len_y_train = collate_fn([torch.from_numpy(x).float() for x in y_train])
    padded_y_test, seq_len_y_test = collate_fn([torch.from_numpy(x).float() for x in y_test])


    inputDim = X_train[0].shape[1]  # takes variable 'x'
    learningRate = 0.1
    epochs = 100
    batch_size = 2000
    num_instance = len(X_train)
    num_batch = math.ceil(num_instance / batch_size)

    model = RNNNet(inputDim, hidden_size, num_layers)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.01)



    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(padded_x_train.cuda())
            #labels = Variable(torch.from_numpy(np.array(padded_y_train)).unsqueeze(1).cuda())
            labels = Variable(padded_y_train.cuda())
        else:
            inputs = Variable(padded_x_train)
            #labels = Variable(torch.from_numpy(np.array(padded_y_train)).unsqueeze(1))
            labels = Variable(padded_y_train)
        #print(inputs.shape, labels.shape)
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        for k in range(num_batch):
            loss = 0
            s_idx = k*batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            inputs_batch = inputs[s_idx:e_idx]
            input_seq_len_batch = seq_len_x_train[s_idx:e_idx]
            labels_batch = labels[s_idx:e_idx]

            # get output from the model, given the inputs
            outputs, out_len_seq = model(pack_padded_sequence(inputs_batch, input_seq_len_batch, batch_first=True))


            # get loss for the predicted output
            loss += criterion(outputs, labels_batch)
            if loss != loss:
                print('loss={}'.format(loss))
                print(torch.isnan(outputs).int().sum())
                print(torch.isnan(labels_batch).int().sum())
                print(s_idx, e_idx, k, num_batch)
                print(outputs, labels_batch)
                print(inputs_batch, labels_batch)
                break
            # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        #print('epoch {}, loss {}'.format(epoch, loss.item()))


    with torch.no_grad():  # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            y_pred, y_pred_out_len_seq = model(pack_padded_sequence(Variable(padded_x_test.cuda()), seq_len_x_test, batch_first=True))
            y_pred = torch.gather(y_pred, 1, torch.unsqueeze(y_pred_out_len_seq - 1, 1).cuda())
            y_pred = y_pred.cpu().data.numpy()
            y_true = torch.gather(padded_y_test.cuda(), 1, torch.unsqueeze(torch.tensor(seq_len_y_test, dtype=torch.int64) - 1, 1).cuda())

        else:
            y_pred, y_pred_out_len_seq = model(pack_padded_sequence(Variable(padded_x_test), seq_len_x_test, batch_first=True))
            y_pred = torch.gather(y_pred, 1, torch.unsqueeze(y_pred_out_len_seq-1, 1))
            y_pred = y_pred.data.numpy()
            y_true = torch.gather(padded_y_test, 1, torch.unsqueeze(torch.tensor(seq_len_y_test, dtype=torch.int64) - 1, 1))

        #y_true = torch.gather(padded_y_test, 1, torch.unsqueeze(torch.Tensor(seq_len_y_test) - 1, 1))
        y_true = y_true.cpu().data.numpy()

        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)

        print('RNN----(t={}, l={})------'.format(test_size, num_layers))
        print('RMSE: {}, MAPE: {}'.format(rmse, mape))
        f.write('RNN---(t={}, l={})--------\n'.format(test_size, num_layers))
        f.write('RMSE: {}, MAPE: {}\n'.format(rmse, mape))
    return



def LSTM_FCN(x, y, test_size, hidden_size=8, num_layers=3):
    x = [i for i in x if i.shape[0] > 2]
    y = [i for i in y if i.shape[0] > 2]

    X_train = [i[:-1] for i in x]
    y_train = [i[:-1] for i in y]
    X_test = x
    y_test = y

    mean = sum([x.mean(axis=0) for x in X_train]) / len(X_train)
    X_train = [x - mean for x in X_train]  # 等价于 train_data = train_data - mean
    std = sum([x.std(axis=0) for x in X_train]) / len(X_train)

    X_train = [(x / (1 + std)) for x in X_train]

    X_test = [x - mean for x in X_test]  # 训练集的均值和标准差
    X_test = [(x / (1 + std)) for x in X_test]

    #X_train = [torch.from_numpy(x).cuda() for x in X_train]


    #padded_x_train, seq_len_x_train = collate_fn([torch.from_numpy(x).float() for x in X_train])
    #padded_x_test, seq_len_x_test = collate_fn([torch.from_numpy(x).float() for x in X_test])

    #padded_y_train, seq_len_y_train = collate_fn([torch.from_numpy(x).float() for x in y_train])
    #padded_y_test, seq_len_y_test = collate_fn([torch.from_numpy(x).float() for x in y_test])

    inputDim = X_train[0].shape[1]  # takes variable 'x'
    learningRate = 0.01
    epochs = 5
    batch_size = 2000
    num_instance = len(X_train)
    num_batch = math.ceil(num_instance / batch_size)

    model = LSTMFCN(inputDim, hidden_size, num_layers)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.01)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs))
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = [Variable(torch.from_numpy(x).cuda()) for x in X_train]
            # labels = Variable(torch.from_numpy(np.array(padded_y_train)).unsqueeze(1).cuda())
            labels = [Variable(torch.from_numpy(x).cuda()) for x in y_train]
        else:
            inputs = [Variable(torch.from_numpy(x)) for x in X_train]
            # labels = Variable(torch.from_numpy(np.array(padded_y_train)).unsqueeze(1))
            labels = [Variable(torch.from_numpy(x)) for x in y_train]
        # print(inputs.shape, labels.shape)
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        for k in range(num_batch):
            print('Batch {}/{}'.format(k, num_batch))
            loss = 0
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            inputs_batch = inputs[s_idx:e_idx]
            #input_seq_len_batch = seq_len_x_train[s_idx:e_idx]
            labels_batch = labels[s_idx:e_idx]

            # get output from the model, given the inputs
            outputs = [model(x) for x in inputs_batch]

            # get loss for the predicted output
            for i in range(len(outputs)):
                #print(outputs[i].shape, labels_batch[i].shape)
                loss += criterion(outputs[i], labels_batch[i])
            if loss != loss:
                print('loss={}'.format(loss))
                print(torch.isnan(outputs).int().sum())
                print(torch.isnan(labels_batch).int().sum())
                print(s_idx, e_idx, k, num_batch)
                print(outputs, labels_batch)
                print(inputs_batch, labels_batch)
                break
            # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        # print('epoch {}, loss {}'.format(epoch, loss.item()))

    with torch.no_grad():  # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            y_pred = [model(x).cpu().data.numpy() for x in [Variable(torch.from_numpy(x).cuda()) for x in X_test]]
            y_true = y_test
            '''
            y_pred, y_pred_out_len_seq = model(
                pack_padded_sequence(Variable(padded_x_test.cuda()), seq_len_x_test, batch_first=True))
            y_pred = torch.gather(y_pred, 1, torch.unsqueeze(y_pred_out_len_seq - 1, 1).cuda())
            y_pred = y_pred.cpu().data.numpy()
            y_true = torch.gather(padded_y_test.cuda(), 1,
                                  torch.unsqueeze(torch.tensor(seq_len_y_test, dtype=torch.int64) - 1, 1).cuda())
            '''

        else:
            y_pred = [model(x).cpu().data.numpy() for x in [Variable(torch.from_numpy(x)) for x in X_test]]
            y_true = y_test
            '''
            y_pred, y_pred_out_len_seq = model(
                pack_padded_sequence(Variable(padded_x_test), seq_len_x_test, batch_first=True))
            y_pred = torch.gather(y_pred, 1, torch.unsqueeze(y_pred_out_len_seq - 1, 1))
            y_pred = y_pred.data.numpy()
            y_true = torch.gather(padded_y_test, 1,
                                  torch.unsqueeze(torch.tensor(seq_len_y_test, dtype=torch.int64) - 1, 1))
            '''

        # y_true = torch.gather(padded_y_test, 1, torch.unsqueeze(torch.Tensor(seq_len_y_test) - 1, 1))
        #y_true = y_true.cpu().data.numpy()
        rmse = 0.0
        mape = 0.0

        n_sample = 0

        for i in range(len(y_pred)):
            #print(y_true[i].shape, y_pred[i].shape)
            rmse += math.sqrt(mean_squared_error(y_true[i], y_pred[i]))
            mape += mean_absolute_percentage_error(y_true[i], y_pred[i])
            n_sample += len(y_true[i])

        rmse /= n_sample
        mape /= n_sample

        print('LSTMFCN----(t={}, l={})------'.format(test_size, num_layers))
        print('RMSE: {}, MAPE: {}'.format(rmse, mape))
        #f.write('RNN---(t={}, l={})--------\n'.format(test_size, num_layers))
        #f.write('RMSE: {}, MAPE: {}\n'.format(rmse, mape))
    return





def main():
    df = pd.concat([
        pd.read_csv(f)
        for f in glob.iglob('results/result_{}_*.csv'.format(PERIOD))], ignore_index=True)
    print(df.info())
    #result = readData(chat, sc, mode=PERIOD)  #mode: d -- day, w -- week, m -- month
    x, y = dataSplit(df)
    #np.save('X_{}_seq.npy'.format(PERIOD), x)
    #np.save('Y_{}_seq.npy'.format(PERIOD), y)
    #np.save('seq_length_{}_seq.npy'.format(PERIOD), seq_length)

    #x = np.load('X_{}.npy'.format(PERIOD))
    #y = np.load('Y_{}.npy'.format(PERIOD))
    t = 0.1
    layer = 2
    test_size = [i / 10 for i in range(1, 6, 1)]
    layers = [1, 2, 3]

    #LSTM_FCN(x, y)

    #for t in test_size:
     #   for layer in layers:
            #LSTM(x, y, test_size=t, num_layers=layer)
            #GRU(x, y, test_size=t, num_layers=layer)
            #RNN(x, y, test_size=t, num_layers=layer)

    LSTM_FCN(x, y, test_size=t, num_layers=layer)



    #for t in test_size:
    #    LSTM(x, y, input_size=x[0].shape[1], test_size=t)

    #print(x[0], y[0], seq_length[0], np.sum(seq_length))

    #print(x.shape, y.shape)
    #plt.scatter(x[:, 0], y)
    #plt.show()

    #result.to_csv('result_{}_{}.csv'.format(PERIOD, START))


if __name__ == "__main__":
    main()
    f.close()



