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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import math


pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)


parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('--start', type=str, default="2021-04", help='Start date(e.g. 2021-04)')
parser.add_argument('--period', type=str, default="m", choices=[
  "d", "w", "m"], help='Period of data separation(day, week, month)')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

PERIOD = args.period
START = args.start

f = open('logs_{}.txt'.format(PERIOD), 'w')


'''
df = pd.concat([
    pd.read_parquet(f)
    for f in iglob('../input/vtuber-livechat/superchats_*.parquet')], ignore_index=True)
'''


def dataSplit(df):
    df['target'] = df.groupby(['channelId'])['totalSC'].shift(-1)
    df = df.query('target == target')
    x = df.drop(['date', 'channelId', 'target', 'Unnamed: 0'], axis=1).astype('float32')
    y = df['target'].astype('float32')
    df.to_csv('tmp.csv')

    return x.to_numpy(), y.to_numpy()


def LR(x, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    score = reg.score(X_train, y_train)
    y_pred = reg.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print('Linear Regression---(t={})--------'.format(test_size))
    print('RMSE: {}, MAPE: {}, R^2: {}'.format(rmse, mape, score))
    f.write('Linear Regression---(t={})--------\n'.format(test_size))
    f.write('RMSE: {}, MAPE: {}, R^2: {}\n'.format(rmse, mape, score))

def Rdg(x, y, test_size, alpha):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    reg = Ridge(alpha=alpha).fit(X_train, y_train)
    score = reg.score(X_train, y_train)
    y_pred = reg.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print('Ridge----(t={}, a={})------'.format(test_size, alpha))
    print('RMSE: {}, MAPE: {}, R^2: {}'.format(rmse, mape, score))
    f.write('Ridge---(t={})--------\n'.format(test_size))
    f.write('RMSE: {}, MAPE: {}, R^2: {}\n'.format(rmse, mape, score))

def Lso(x, y, test_size, alpha):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    reg = Lasso(alpha=alpha).fit(X_train, y_train)
    score = reg.score(X_train, y_train)
    y_pred = reg.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print('Lasso----(t={}, a={})------'.format(test_size, alpha))
    print('RMSE: {}, MAPE: {}, R^2: {}'.format(rmse, mape, score))
    f.write('Lasso---(t={})--------\n'.format(test_size))
    f.write('RMSE: {}, MAPE: {}, R^2: {}\n'.format(rmse, mape, score))

def SGD(x, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3)).fit(X_train, y_train)
    score = reg.score(X_train, y_train)
    y_pred = reg.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print('SGD----(t={})------'.format(test_size))
    print('RMSE: {}, MAPE: {}, R^2: {}'.format(rmse, mape, score))
    f.write('SGD---(t={})--------\n'.format(test_size))
    f.write('RMSE: {}, MAPE: {}, R^2: {}\n'.format(rmse, mape, score))

def GBR(x, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    reg = GradientBoostingRegressor(random_state=0).fit(X_train, y_train)
    score = reg.score(X_train, y_train)
    y_pred = reg.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print('GBR----(t={})------'.format(test_size))
    print('RMSE: {}, MAPE: {}, R^2: {}'.format(rmse, mape, score))
    f.write('GBR---(t={})--------\n'.format(test_size))
    f.write('RMSE: {}, MAPE: {}, R^2: {}\n'.format(rmse, mape, score))

def XGBoost(x, y, test_size, es=1000, depth=7):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    model = xgboost.XGBRegressor(n_estimators=es, max_depth=depth, eta=0.1, subsample=0.7,
                                 colsample_bytree=0.8).fit(X_train, y_train)
    score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print('XGBoost----(t={}, es={}, depth={})------'.format(test_size, es, depth))
    print('RMSE: {}, MAPE: {}, R^2: {}'.format(rmse, mape, score))
    f.write('XGBoost---(t={})--------\n'.format(test_size))
    f.write('RMSE: {}, MAPE: {}, R^2: {}\n'.format(rmse, mape, score))


def NN(x, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    mean = X_train.mean(axis=0)
    X_train -= mean  # 等价于 train_data = train_data - mean
    std = X_train.std(axis=0)
    print(mean, std)
    X_train / (1 + std)



    X_test -= mean  # 训练集的均值和标准差
    X_test /= (1 + std)

    inputDim = X_train.shape[1]  # takes variable 'x'
    outputDim = 1 # takes variable 'y'
    learningRate = 1
    epochs = 100
    batch_size = 2000
    num_instance = X_train.shape[0]
    num_batch = math.ceil(num_instance / batch_size)


    class linearRegression(torch.nn.Module):
        def __init__(self, inputSize, outputSize, hidden=4):
            super(linearRegression, self).__init__()
            self.linear1 = torch.nn.Linear(inputSize, hidden)
            self.linear2 = torch.nn.Linear(hidden, outputSize)
            self.batch_norm = torch.nn.BatchNorm1d(hidden)
            self.dropout = torch.nn.Dropout(0.2)

        def forward(self, x):
            x = self.linear1(x)
            x = self.batch_norm(x)
            x = self.dropout(x)
            out = self.linear2(x)
            return out


    model = linearRegression(inputDim, outputDim)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.01)



    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(X_train).cuda())
            labels = Variable(torch.from_numpy(y_train).unsqueeze(1).cuda())
        else:
            inputs = Variable(torch.from_numpy(X_train))
            labels = Variable(torch.from_numpy(y_train).unsqueeze(1))
        #print(inputs.shape, labels.shape)
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        for k in range(num_batch):
            loss = 0
            s_idx = k*batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            inputs_batch = inputs[s_idx:e_idx]
            labels_batch = labels[s_idx:e_idx]

            # get output from the model, given the inputs
            outputs = model(inputs_batch)

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
            y_pred = model(Variable(torch.from_numpy(X_test).cuda())).cpu().data.numpy()
        else:
            y_pred = model(Variable(torch.from_numpy(X_test))).data.numpy()

        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print('NN----(t={})------'.format(test_size))
        print('RMSE: {}, MAPE: {}'.format(rmse, mape))
        f.write('NN---(t={})--------\n'.format(test_size))
        f.write('RMSE: {}, MAPE: {}\n'.format(rmse, mape))


def main():
    df = pd.concat([
        pd.read_csv(f)
        for f in glob.iglob('result_{}_*.csv'.format(PERIOD))], ignore_index=True)
    print(df.info())
    #result = readData(chat, sc, mode=PERIOD)  #mode: d -- day, w -- week, m -- month
    x, y = dataSplit(df)
    np.save('X_{}.npy'.format(PERIOD), x)
    np.save('Y_{}.npy'.format(PERIOD), y)
    alphas = [0, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000, 10000, 100000]
    test_size = [i / 10 for i in range(1, 6, 1)]
    estimators = [i*1000 for i in range(1, 6, 1)]
    depths = [i+6 for i in range(1, 6, 2)]
    x = np.load('X_{}.npy'.format(PERIOD))[: , :-7]
    y = np.load('Y_{}.npy'.format(PERIOD))

    mean = x.mean(axis=0)
    x -= mean  # 等价于 train_data = train_data - mean
    std = x.std(axis=0)
    x / (1 + std)
    print(mean, std)

    mean = y.mean(axis=0)
    y -= mean  # 等价于 train_data = train_data - mean
    std = y.std(axis=0)
    y / (1 + std)
    print(mean, std)

    #print(x.shape, y.shape)
    #plt.scatter(x[:, 0], y)
    #plt.show()

    t = 0.1
    a = 1
    LR(x, y, t)
    SGD(x, y, t)
    GBR(x, y, t)
    XGBoost(x, y, t)
    NN(x, y, t)
    Rdg(x, y, t, a)
    Lso(x, y, t, a)
    '''
    for t in test_size:
        LR(x, y, t)

    for t in test_size:
        SGD(x, y, t)

    for t in test_size:
        GBR(x, y, t)

    for t in test_size:
        for a in alphas:
            Rdg(x, y, t, a)

    for t in test_size:
        for a in alphas:
            Lso(x, y, t, a)

    for t in test_size:
        for es in estimators:
            for dep in depths:
                XGBoost(x, y, t, es, dep)

    for t in test_size:
        NN(x, y, t)
    '''
    #result.to_csv('result_{}_{}.csv'.format(PERIOD, START))


if __name__ == "__main__":
    main()
    f.close()



