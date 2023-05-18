import pandas as pd
import numpy as np
import glob
import scipy.sparse as sp
from preprocess import target
import re

'''
date_concat = ['2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
               '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07']
'''
date_concat = ['2021-04']

period = 'm'

pd.set_option('display.max_columns', None)


def DF2Adj(nodeList, df):
    tmp = df.reindex(nodeList, axis='columns', fill_value=0.0)
    result = tmp.reindex(nodeList, axis='rows', fill_value=0.0)

    return sp.csr_matrix(result.to_numpy(), dtype=np.float32)

def DF2Adj_nodeFeature(nodeList, node_feature_df, date):
    df = node_feature_df.query('date == @date')
    tmp =df.drop(['date'], axis=1).set_index('channelId')
    #nodes = df.index.to_list()
    result = tmp.reindex(nodeList, axis='rows', fill_value=0.0)

    result = result.drop(['impact1', 'impact2', 'impact3', 'impact4', 'impact5', 'impact6', 'impact7'], axis=1)

    #print(result)

    return sp.csr_matrix(result.to_numpy(), dtype=np.float32)



def readData(period):
    #result = readData(chat, sc, mode=PERIOD)  #mode: d -- day, w -- week, m -- month
    #result = pd.read_csv('result_{}.csv'.format(date_concat))

    if period == 'm':
        date_concat = [re.findall(r'overlaps\\overlap_viewers_(\d{4}-\d{2}).csv', f)[0]
                       for f in glob.glob('overlaps/overlap_viewers_*.csv')
                       if re.match(r'overlaps\\overlap_viewers_\d{4}-\d{2}.csv', f)]
    elif period == 'w':
        date_concat = [re.findall(r'overlaps\\overlap_viewers_(\d{4}-\d{2}-\d{1}).csv', f)[0]
                       for f in glob.glob('overlaps/overlap_viewers_*.csv')
                       if re.match(r'overlaps\\overlap_viewers_\d{4}-\d{2}-\d{1}.csv', f)]

    #date_concat = ['2021-09', '2021-10', '2021-11']

    label_concat = [re.findall(r'results\\result_[mwd]_(\d{4}-\d{2}).csv', f)[0]
                       for f in glob.glob('results\\result_{}_*.csv'.format(period))
                       if re.match(r'results\\result_[mwd]_(\d{4}-\d{2}).csv', f)]

    adj_viewer = dict()
    adj_period = dict()
    node_feature = dict()
    nodes = dict()

    print(label_concat)
    print(date_concat)
    labels, node_feature_df = target(date_concat, period, label_concat)
    channels = pd.read_csv('Vtuber1B_elements/channels.csv')
    node_feature_df = pd.merge(node_feature_df, channels[['channelId', 'subscriptionCount', 'videoCount']], how='left',
                  on='channelId')

    #dateList = labels['date'].drop_duplicates().tolist()


    overlap_description = pd.read_csv('overlaps/overlap_description.csv', index_col=0)  # description
    nodeList = overlap_description.index.tolist()
    #print(node_feature_df.head(5))

    for date in date_concat:
        node_feature[date] = DF2Adj_nodeFeature(nodeList, node_feature_df, date)
        # viewer                                                              , index_col=0))
        adj_viewer[date] = DF2Adj(nodeList, pd.read_csv('overlaps/overlap_viewers_{}.csv'.format(date), index_col=0))
        # period
        adj_period[date] = DF2Adj(nodeList, pd.read_csv('overlaps/overlap_period_{}.csv'.format(date), index_col=0))
        node_tmp = labels.query('date == @date')['channelId'].drop_duplicates().tolist()
        nodes[date] = [x for x in node_tmp if x in nodeList]


    adj_description = overlap_description.to_numpy()

    #print(dateList)

    return date_concat[:-1], node_feature, adj_viewer, adj_period, adj_description, labels, nodes, nodeList

f = glob.glob('overlaps/overlap_viewers_*.csv')


'''
print(f)

if period == 'm':
    date_concat = [re.findall(r'overlaps\\overlap_viewers_(\d{4}-\d{2}).csv', f)
                   for f in glob.glob('overlaps/overlap_viewers_*.csv')
                   if re.match(r'overlaps\\overlap_viewers_\d{4}-\d{2}.csv', f)]
elif period == 'w':
    date_concat = [re.findall(r'overlaps\\overlap_viewers_(\d{4}-\d{2}-\d{1}).csv', f)
                   for f in glob.glob('overlaps/overlap_viewers_*.csv')
                   if re.match(r'overlaps\\overlap_viewers_\d{4}-\d{2}-\d{1}.csv', f)]
print(date_concat)

if period == 'm':
    date_concat = [re.findall(r'overlaps\\overlap_period_(\d{4}-\d{2}).csv', f)
                   for f in glob.glob('overlaps/overlap_period_*.csv')
                   if re.match(r'overlaps\\overlap_period_\d{4}-\d{2}.csv', f)]
elif period == 'w':
    date_concat = [re.findall(r'overlaps\\overlap_period_(\d{4}-\d{2}-\d{1}).csv', f)
                   for f in glob.glob('overlaps/overlap_period_*.csv')
                   if re.match(r'overlaps\\overlap_period_\d{4}-\d{2}-\d{1}.csv', f)]

print(date_concat)
'''

if __name__ == "__main__":
    readData(period)
