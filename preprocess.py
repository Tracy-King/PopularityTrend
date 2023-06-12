import pandas as pd
import numpy as np
import glob
import torch
import argparse
from tqdm import tqdm
import sys
import scipy.sparse as sp
import os

parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('--start', type=str, default="2021-04", help='Start date(e.g. 2021-04)')
parser.add_argument('--period', type=str, default="m", choices=[
  "d", "w", "m"], help='Period of data separation(day, week, month)')
parser.add_argument('--year', type=str, default="2021", choices=["2021", "2022"],
                    help='Period of data separation(day, week, month)')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

PERIOD = args.period
START = args.start
YEAR = args.year




def dataGeneration(dateList, df, sc_df):
    result = pd.DataFrame(columns=['date', 'channelId', 'chats', 'memberChats', 'uniqueChatters', 'uniqueMembers',
                               'superChats', 'uniqueSuperChatters', 'totalSC', 'totalLength', 'impact1', 'impact2',
                               'impact3', 'impact4', 'impact5', 'impact6', 'impact7'])

    currencyRate = {'JPY':1.0, 'USD':134.84, 'TWD':4.43, 'KRW':0.10, 'PHP':2.44, 'HDK':17.18, 'CAD':99.89, 'EUR':143.41,
                'SGD':100.66, 'GBP':161.01, 'AUD':92.20, 'RUB':1.79, 'MXN':7.25, 'BRL':25.84, 'ARS':0.70,
                'CLP':0.17, 'SEK':12.84, 'INR':1.63, 'PLN':30.05, 'NZD':83.82, 'PEN':34.82, 'NOK':13.09,
                'DKK':19.26, 'HUF':0.37, 'CHF':145.05, 'CZK':6.06, 'HRK':19.03, 'COP':0.02, 'BGN':73.36,
                'HNL':5.44, 'ISK':0.93, 'CRC':0.24, 'RON':29.26, 'ZAR':7.40, 'GTQ':17.15, 'BYN':53.14,
                'RSD':1.22, 'BOB':19.43, 'UYU':3.43, 'DOP':2.39, 'MKD':2.33}

    for date in dateList:
        tmp = df.query('timestamp == @date') #df[df['timestamp'].isin([date])]
        tmp_sc = sc_df.query('timestamp == @date') # sc_df[df['timestamp'].isin([date])]
        unique_tmp = tmp.drop_duplicates(subset=['authorChannelId', 'channelId'], keep='first')
        unique_tmp_sc = tmp_sc.drop_duplicates(subset=['authorChannelId', 'channelId'], keep='first')

        for index, row in tmp_sc.iterrows():
            for k, v in currencyRate.items():
                if row['currency'] == k:
                    row['amount'] = row['amount']*v

        channelId = tmp['channelId'].drop_duplicates().tolist()
        chats = tmp['channelId'].value_counts().to_dict()
        memberChats = tmp[tmp['isMember'].isin([1.0])]['channelId'].value_counts().to_dict()
        uniqueChatters = unique_tmp['channelId'].value_counts().to_dict()
        uniqueMembers = unique_tmp[tmp['isMember'].isin([1.0])]['channelId'].value_counts().to_dict()

        superChats = tmp_sc['channelId'].value_counts().to_dict()
        uniqueSuperChatters = unique_tmp_sc['channelId'].value_counts().to_dict()
        totalLength = tmp.groupby(['channelId'])['bodyLength'].sum().to_dict()
        totalSC = tmp_sc.groupby(['channelId'])['amount'].sum().to_dict()
        impact1 = tmp_sc[tmp_sc['significance'].isin([1])].groupby(['channelId'])['significance'].count().to_dict()
        impact2 = tmp_sc[tmp_sc['significance'].isin([2])].groupby(['channelId'])['significance'].count().to_dict()
        impact3 = tmp_sc[tmp_sc['significance'].isin([5])].groupby(['channelId'])['significance'].count().to_dict()
        impact4 = tmp_sc[tmp_sc['significance'].isin([10])].groupby(['channelId'])['significance'].count().to_dict()
        impact5 = tmp_sc[tmp_sc['significance'].isin([20])].groupby(['channelId'])['significance'].count().to_dict()
        impact6 = tmp_sc[tmp_sc['significance'].isin([50])].groupby(['channelId'])['significance'].count().to_dict()
        impact7 = tmp_sc[tmp_sc['significance'].isin([100])].groupby(['channelId'])['significance'].count().to_dict()

        for id in channelId:
            result.loc[len(result)] = [date, id, chats.get(id, 0), memberChats.get(id, 0), uniqueChatters.get(id, 0),
                                   uniqueMembers.get(id, 0), superChats.get(id, 0), uniqueSuperChatters.get(id, 0),
                                   totalLength.get(id, 0), totalSC.get(id, 0), impact1.get(id, 0), impact2.get(id, 0),
                                   impact3.get(id, 0), impact4.get(id, 0), impact5.get(id, 0), impact6.get(id, 0),
                                   impact7.get(id, 0)]

        print('Date {} finished.'.format(date))
        break

    return result


def dataProcess(chat_f, sc_f, mode='d'):
    df = pd.read_parquet(chat_f)
    sc_df = pd.read_parquet(sc_f)

    sc_df['impact'] = sc_df['significance'].map({
        1: 1,
        2: 2,
        3: 5,
        4: 10,
        5: 20,
        6: 50,
        7: 100})

    channels = pd.read_csv('Vtuber1B_elements/channels.csv')
    df = pd.merge(df, channels, how='left', on='channelId')
    sc_df = pd.merge(sc_df, channels, how='left', on='channelId')

    df["timestamp"] = pd.to_datetime(df['timestamp'])
    df.sort_index(inplace=False)
    df["timestamp"] = df["timestamp"].dt.to_period(mode)

    sc_df["timestamp"] = pd.to_datetime(sc_df['timestamp'])
    sc_df.sort_index(inplace=False)
    sc_df["timestamp"] = sc_df["timestamp"].dt.to_period(mode)


    dateList = df['timestamp'].drop_duplicates().tolist()

    #print(dateList)

    result = dataGeneration(dateList, df, sc_df)

    return result.convert_dtypes()


def duplicated_viewers(df):
    channelId = df['channelId'].unique().tolist()
    df_overlap = pd.DataFrame(columns=channelId)
    viewer_dict = dict()
    for i in tqdm(range(len(channelId))):
        bucket = df.loc[df['channelId'] == channelId[i]]
        viewerList = bucket['authorChannelId'].unique().tolist()
        viewer_dict[channelId[i]] = viewerList


    for i in tqdm(range(len(channelId))):
        this_column = df_overlap.columns[i]
        Left = channelId[i]
        value = []
        for x in channelId:
            Bucket1 = viewer_dict[Left]
            left_len = len(Bucket1)

            Right = x
            Bucket2 = viewer_dict[Right]
            right_len = len(Bucket2)

            Intersect = set(Bucket1).intersection(Bucket2)
            if left_len == 0:
                value.append(0)
            else:
                value.append((len(Intersect) * 100) /left_len)

        df_overlap[this_column] = value

        # print("column "+ str(i+1) + " from all " + str(len(englishName)) +" is done")
    df_overlap.index = channelId
    #fig, ax = plt.subplots(figsize=(32, 32))

    #df_overlap.to_csv('overlap_viewer.csv')

    #sns.heatmap(df_overlap, annot=True)

    return df_overlap



def duplicated_period(df, date):
    channelId = df['channelId'].unique().tolist()
    df_overlap = pd.DataFrame(columns=channelId)
    value = []
    period_dict = dict()
    for i in tqdm(range(len(channelId))):
        bucket = df.loc[df['channelId'] == channelId[i]] ## 计算每个channel的直播区间并加到dict里
        ts = pd.to_datetime(bucket['timestamp'])
        ts.sort_index(inplace=False)
        ts = ts.dt.to_period('h')
        tsList = ts.drop_duplicates().tolist()
        #print(tsList)       ## 该channel的直播区间（yyyy-mm-dd hh）
        period_dict[channelId[i]] = tsList

    '''
    for i in tqdm(range(len(channelId))):
        this_column = df_overlap.columns[i]
        Left = channelId[i]
        value = []
        for x in channelId:
            Bucket1 = period_dict[Left]
            left_len = len(Bucket1)

            Right = x
            Bucket2 = period_dict[Right]
            right_len = len(Bucket2)
            #Right_Viewer = Bucket2['authorChannelId'].unique().tolist()

            Intersect = set(Bucket1).intersection(Bucket2)
            if left_len == 0:
                value.append(0)
            else:
                value.append((len(Intersect) * 100) / left_len)

        df_overlap[this_column] = value

        # print("column "+ str(i+1) + " from all " + str(len(englishName)) +" is done")
    df_overlap.index = channelId
    #fig, ax = plt.subplots(figsize=(32, 32))
    '''
    data = []
    for k, v in period_dict.items():
        data.append([date, k, len(v)])

    df_hrs = pd.DataFrame(data, columns=['date', 'channelId', 'hrs'])
    print(df_hrs.info(), df_hrs.head(5))

    #sns.heatmap(df_overlap, annot=True)
    return df_overlap, df_hrs


def duplicated_description():
    df = pd.read_csv('Vtuber1B_elements/channels.csv')
    #df = pd.merge(df, channels, how='left', on='channelId')
    affiliation = df['affiliation']
    group = df['group']

    channelId = df['channelId'].unique().tolist()
    df_overlap = pd.DataFrame(columns=channelId)

    for i in tqdm(range(len(channelId))):
        this_column = df_overlap.columns[i]
        Left = channelId[i]
        value = []
        leftAffi = df.query('channelId == @Left')['affiliation'].tolist()[0]
        leftgrp = df.query('channelId == @Left')['group'].tolist()[0]
        if leftAffi is None or leftgrp is None:
            df_overlap[this_column] = [0]*len(channelId)
            continue
        for x in channelId:
            Right = x
            rightAffi = df.query('channelId == @Right')['affiliation'].tolist()[0]
            rightgrp = df.query('channelId == @Right')['group'].tolist()[0]
            tmp = 0
            if leftAffi == rightAffi or leftgrp == rightgrp:
                tmp = 1
            if leftAffi == rightAffi and leftgrp == rightgrp:
                tmp = 2
            if not (leftAffi == rightAffi or leftgrp == rightgrp):
                tmp = 0
            value.append(tmp)

        df_overlap[this_column] = value

        # print("column "+ str(i+1) + " from all " + str(len(englishName)) +" is done")
    df_overlap.index = channelId
    # fig, ax = plt.subplots(figsize=(32, 32))

    # sns.heatmap(df_overlap, annot=True)
    return df_overlap

def channelFeatures(df):
    nodeFeature = pd.read_csv('results/result_{}_{}.csv'.format(PERIOD, START))

    return nodeFeature

def target(date_concat, p, label_concat):

    if os.path.isfile('label_{}.csv'.format(p)) and os.path.isfile('node_features_{}.csv'.format(p)):
        target = pd.read_csv('label_{}.csv'.format(p), index_col=0)
        node_features = pd.read_csv('node_features_{}.csv'.format(p), index_col=0)
        return target, node_features



    if p == 'm':
        df = pd.concat([
            pd.read_csv('results/result_{}_{}.csv'.format(p, date), index_col=0)
            for date in label_concat], ignore_index=True)

    elif p =='w':
        c_list = []
        for date in label_concat:
            tmp = pd.read_csv('results/result_{}_{}.csv'.format(p, date), index_col=0)
            dates = tmp['date'].drop_duplicates().tolist()
            filenames = [date+'-{}'.format(i) for i in range(len(dates))]

            sort_map = pd.DataFrame({'date': dates, 'filename': filenames}).reset_index().set_index('date')
            tmp['filename'] = tmp['date'].map(sort_map['filename'])
            df = df.drop(['date'], axis=1)
            df = df.rename(columns={'filename': 'date'})
            #print(sort_map)
            #print(tmp.head(5))
            c_list.append(tmp)

        df = pd.concat(c_list, ignore_index=True)


    df['target'] = df.groupby(['channelId'])['totalSC'].shift(-1)
    #nodelist = df['channelId'].unique()
    df = df.query('target == target')
    df['perf'] = df['target'] / df['totalSC']


    target = df[['date', 'channelId', 'target', 'perf']]

    node_features = df.drop(['target', 'perf'], axis=1)
    '''
    for date in date_concat:
        tmp_df = node_features.query('date == @date')
        duplicate_channels = tmp_df.duplicated(subset=['channelId'], keep=False)
        #print(duplicate_channels)
        #print(tmp_df[duplicate_channels])
    '''
    target.to_csv('label_{}.csv'.format(p))
    node_features.to_csv('node_features_{}.csv'.format(p))

    return target, node_features


if __name__ == "__main__":

    for f in glob.glob('parquets/chats_{}-*.parquet'.format(YEAR)):
        print(f)

        #chat = pd.read_parquet('chats_{}.parquet'.format('2021-04'))
        chat = pd.read_parquet(f)

        if PERIOD == 'w':
            chat['period'] = pd.to_datetime(chat['timestamp'])
            chat['period'] = chat['period'].dt.to_period(PERIOD)
            df_group = chat.groupby(['period'])
            dateList = chat['period'].drop_duplicates().tolist()
            for i in range(len(dateList)):
                print(dateList[i])
                date = dateList[i]
                df, df_hrs = duplicated_period(df_group.get_group(date), date)
                #df.to_csv('overlap_period_{}-{}.csv'.format(f[6:13], i))
                df_hrs.to_csv('hrs_{}-{}.csv'.format(f[6:13], i))
                #df = duplicated_viewers(df_group.get_group(date))
                #df.to_csv('overlap_viewers_{}-{}.csv'.format(f[6:13], i))
        else:
            chat['period'] = pd.to_datetime(chat['timestamp'])
            date = chat['period'].drop_duplicates().tolist()[0]
            df, df_hrs = duplicated_period(chat, date)
            #df.to_csv('overlap_period_{}.csv'.format(f[6:13]))
            df_hrs.to_csv('hrs_{}-{}.csv'.format(f[6:13], i))
            #df = duplicated_viewers(chat)
            #df.to_csv('overlap_viewers_{}.csv'.format(f[6:13]))

    #df = duplicated_description()
    #df.to_csv('overlap_description.csv')
