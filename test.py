import pandas as pd
import numpy as np
import glob
import matplotlib as plt
#from plotly.subplots import make_subplot
from datetime import datetime
import argparse
from tqdm import tqdm


pd.set_option('display.max_columns', None)


parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('--start', type=str, default="2021-04", help='Start date(e.g. 2021-04)')
parser.add_argument('--period', type=str, default="w", choices=[
  "d", "w", "m"], help='Period of data separation(day, week, month)')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

PERIOD = args.period
START = args.start


'''
df = pd.concat([
    pd.read_parquet(f)
    for f in iglob('../input/vtuber-livechat/superchats_*.parquet')], ignore_index=True)
'''


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

        '''
        print(chats)
        print(memberChats)
        print(uniqueChatters)
        print(uniqueMembers)
        print(superChats)
        print(uniqueSuperChatters)
        print(totalLength)
        print(totalSC)
        print(impact1)
        print(impact2)
        print(impact3)
        print(impact4)
        print(impact5)
        print(impact6)
        print(impact7)
        #print(superChats, uniqueSuperChatters, totalLength, totalSC)
        #print(impact1, impact2, impact3, impact4, impact5, impact6, impact7)
        '''

        for id in channelId:
            result.loc[len(result)] = [date, id, chats.get(id, 0), memberChats.get(id, 0), uniqueChatters.get(id, 0),
                                   uniqueMembers.get(id, 0), superChats.get(id, 0), uniqueSuperChatters.get(id, 0),
                                   totalLength.get(id, 0), totalSC.get(id, 0), impact1.get(id, 0), impact2.get(id, 0),
                                   impact3.get(id, 0), impact4.get(id, 0), impact5.get(id, 0), impact6.get(id, 0),
                                   impact7.get(id, 0)]

        print('Date {} finished.'.format(date))
        break

    return result




def readData(chat_f, sc_f, mode='d'):
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


def main():
    chat = 'chats_{}.parquet'.format(START)
    sc = 'superchats_{}.parquet'.format(START)

    result = readData(chat, sc, mode=PERIOD)  #mode: d -- day, w -- week, m -- month

    result.to_csv('result_{}_{}.csv'.format(PERIOD, START))


if __name__ == "__main__":
    #main()
    df = pd.read_csv('overlap_period.csv')
    print(df)
    print(df.info())



