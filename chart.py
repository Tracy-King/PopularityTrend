import pandas as pd
import numpy as np
import glob
import sys
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
# from plotly.subplots import make_subplot
from datetime import datetime
import argparse
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline import init_notebook_mode, iplot
pio.renderers.default = "browser"

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


'''
df = pd.concat([
    pd.read_parquet(f)
    for f in iglob('../input/vtuber-livechat/superchats_*.parquet')], ignore_index=True)
'''


def dataSplit(df):
    df['target'] = df.groupby(['channelId'])['totalSC'].shift(-1)
    df = df.query('target == target')
    dfgroup = df.groupby(['channelId'])
    channelList = df['channelId'].drop_duplicates().tolist()
    # dateList = df['date'].drop_duplicates().tolist()

    x = []
    y = []
    seq_length = []

    for channel in channelList:
        tmp = dfgroup.get_group(channel).sort_values(by=['date'])
        # print(tmp)
        x_ = tmp.drop(['date', 'channelId', 'target', 'Unnamed: 0'], axis=1).astype('float32').to_numpy()
        y_ = tmp['target'].astype('float32').to_numpy()[-1]
        length = tmp.shape[0]
        # print(x_, y_, length)
        # print('___________________________')

        x.append(x_)
        y.append(y_)
        # print(x_.shape, y_.shape, length)
        # print('channel {} finished.'.format(channel))

    # df.to_csv('tmp.csv')

    return x, y


def main():
    df = pd.concat([
        pd.read_csv(f)
        for f in glob.iglob('result_{}_*.csv'.format(PERIOD))], ignore_index=True)
    print(df.info())
    channels = pd.read_csv('Vtuber1B_elements/channels.csv')
    top_sc_list = ['UCb5JxV6vKlYVknoJB8TnyYg', 'UCIBY1ollUsauvVi4hW4cumw', 'UC_vMYWcDjmfdpH6r4TTn1MQ',
                   'UCmZ1Rbthn-6Jm_qOGjYsh5A',
                   'UC-6rZgmxZSIbq786j3RD5ow', 'UCXU7YYxy_iQd3ulXyO-zC2w', 'UCckdfYDGrjojJM28n5SHYrA',
                   'UC6wvdADTJ88OfIbJYIpAaDA', 'UCPvGypSgfDkVe7JG2KygK7A', 'UCo2N7C-Z91waaR6lF3LL_jw']

    # result = readData(chat, sc, mode=PERIOD)  #mode: d -- day, w -- week, m -- month
    #x, y = dataSplit(df)
    #dateList = df['date'].drop_duplicates().tolist()
    dailySC = df.groupby(['date'])['totalSC'].sum().to_dict()
    dailyChannels = df.groupby(['date'])['channelId'].count().to_dict()
    print(dailySC, dailyChannels)

    new_df = pd.DataFrame.from_dict(dailySC, orient='index')
    new_df.rename(columns={0: 'dailySC'}, inplace=True)
    new_df['dailyChannels'] = [dailyChannels[k] for k in new_df.index.tolist()]

    new_df['avgDailySC'] = new_df['dailySC'].div(new_df['dailyChannels'])
    print(new_df)
    mask = (new_df.index > '2021-11-01') & (new_df.index <= '2021-12-31')
    print(new_df.loc[mask])

    fig = px.line(new_df, x=new_df.index, y='avgDailySC')
    fig.update_layout(title='Average SC', xaxis_title='Date', yaxis_title='Amount of income')
    # fig.show()
    fig.write_image('{}_Avg_income.png'.format(PERIOD), engine='orca')


    '''
    for channel in top_sc_list:
        df_group = df.groupby(['channelId'])
        channel_df = df_group.get_group(channel)
        sorted_channel_df = channel_df.sort_values(['date'])
        channel_name = channels.query('channelId == @channel')['englishName'].tolist()[0]
        fig = px.line(sorted_channel_df, x="date", y='totalSC')
        fig.update_layout(title='{}'.format(channel_name), xaxis_title='Date', yaxis_title='Amount of income')
        #fig.show()
        fig.write_image('{}_{}_income.png'.format(PERIOD, channel_name), engine='orca')
        print(channel_name)
    '''












if __name__ == "__main__":
    main()

