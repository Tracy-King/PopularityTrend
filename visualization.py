import pandas as pd
import numpy as np
from glob import iglob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


pd.set_option('display.max_columns', None)
#pd.options.plotting.backend = 'plotly'
import plotly.io as pio
pio.renderers.default = "browser"

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

channels = pd.read_csv('Vtuber1B_elements/channels.csv')

df_superchat_stats = pd.read_csv('Vtuber1B_elements/superchat_stats.csv')
df_superchat_stats = pd.merge(df_superchat_stats, channels, how='left', on='channelId')

df_chat_stats = pd.read_csv('Vtuber1B_elements/chat_stats.csv')
df_chat_stats = pd.merge(df_chat_stats, channels, how='left', on='channelId')

#Graduated and Official Channel
Graduated = ["Kiryu Coco","Uruha Rushia","Hololive VTuber Group"]
df_superchat_stats = df_superchat_stats[~df_superchat_stats.englishName.isin(Graduated)]
df_chat_stats = df_chat_stats[~df_chat_stats.englishName.isin(Graduated)]

df_superchat_stats.info()
df_chat_stats.info()

print(df_superchat_stats.query('period=="2021-03"').nlargest(10, 'totalSC'))

top_sc_list = ['UCb5JxV6vKlYVknoJB8TnyYg', 'UCIBY1ollUsauvVi4hW4cumw', 'UC_vMYWcDjmfdpH6r4TTn1MQ', 'UCmZ1Rbthn-6Jm_qOGjYsh5A',
               'UC-6rZgmxZSIbq786j3RD5ow', 'UCXU7YYxy_iQd3ulXyO-zC2w', 'UCckdfYDGrjojJM28n5SHYrA', 'UC6wvdADTJ88OfIbJYIpAaDA', 'UCPvGypSgfDkVe7JG2KygK7A', 'UCo2N7C-Z91waaR6lF3LL_jw']

top_sc_03 = ['UCHsx4Hqa-1ORjQTh9TYDhww', 'UCFKOVgVbGmX65RxO3EtH3iw', 'UC1DCedRgGHBdm81E1llLhOQ', 'UCL_qhgtOy0dy1Agp8vkySQg',
               'UCCzUftO8KOVkV4wQG1vkUvg', 'UCP9ZgeIJ3Ri9En69R0kJc9Q', 'UC1opHUrw8rvnsadT-iGp7Cg', 'UChAnqc_AY5_I3Px5dig3X1Q', 'UCoSrY_IQQVpmIRZ9Xf-y93g', 'UC5CwaMl1eIgY8h02uZw7u8A']


for channel in top_sc_03:
    ex = df_superchat_stats[df_superchat_stats['channelId'].isin([channel])]
    ex2 = df_chat_stats[df_chat_stats['channelId'].isin([channel])]
    channel_name = ex['name'].tolist()[0]
    # print(name)
    #print(ex)
    fig = px.line(ex2, x="period", y=["chats", 'memberChats'])
    fig.update_layout(title='{}'.format(channel_name), xaxis_title='Period', yaxis_title='# of chats')
    fig.write_image('{}_chats.png'.format(channel_name))

    fig = px.line(ex2, x="period", y=['uniqueChatters', 'uniqueMembers'])
    fig.update_layout(title='{}'.format(channel_name), xaxis_title='Period', yaxis_title='Unique # of chats')
    fig.write_image('{}_uniqueChats.png'.format(channel_name))

    fig = px.line(ex, x="period", y="totalSC")
    fig.update_layout(title='{}'.format(channel_name), xaxis_title='Period', yaxis_title='Amount $ of SC')
    fig.write_image('{}_$ofSC.png'.format(channel_name))

    fig = px.line(ex, x="period", y=["superChats", 'uniqueSuperChatters'])
    fig.update_layout(title='{}'.format(channel_name), xaxis_title='Period', yaxis_title='# of SC')
    fig.write_image('{}_#ofSC.png'.format(channel_name))


#print(df_superchat_stats.nlargest(10, 'totalSC'))
#print(df_chat_stats.nlargest(10, 'chats'))
#df_detailed = df_detailed[~df_detailed.englishName.isin(Graduated)]

#Only choose Hololive
#ff = ["Hololive"]
#df_superchat_stats = df_superchat_stats[df_superchat_stats.affiliation.isin(Aff)]
#df_chat_stats = df_chat_stats[~df_chat_stats.affiliation.isin(Aff)]
#df_detailed = df_detailed[~df_detailed.affiliation.isin(Aff)]




#                ['97DWg8tqo4M', 'sXnTgUkXqEE', 'zl5P5lAvLwM', 'GsgbCSC6d50', 'TDXBiMKQZpI', 'fkWB_8Yyt0A', '8QEhoC-DOjM', 'DaT7j74W7zw', '1kxCz6tt2MU']
# concat_list = ['fkWB_8Yyt0A', '8QEhoC-DOjM', 'DaT7j74W7zw', '1kxCz6tt2MU']

# concat_list = ['ON3WijEIS1c', 'qO8Ld-qLjb0', 'k3Nzow_OqQY', 'y3DCfZmX8iA', 'qHZwDxea7fQ']#, 'cibdBr9TkEo', 'rW8jSXVsW2E', 'eIi8zCPFyng', 'wtJj3CO_YR0']
#                ['97DWg8tqo4M', 'sXnTgUkXqEE', 'zl5P5lAvLwM', 'GsgbCSC6d50', 'TDXBiMKQZpI', 'fkWB_8Yyt0A', '8QEhoC-DOjM', 'DaT7j74W7zw', '1kxCz6tt2MU']
# concat_list = ['fkWB_8Yyt0A', '8QEhoC-DOjM', 'DaT7j74W7zw', '1kxCz6tt2MU']
'''
data = 0  # pd.read_pickle('../dynamicGraph/concat_full_v3_tmp.pkl')
cnt = 1
end_time = 0  # data['Offset'].iat[-1].to_numpy()
for id in concat_list:
    new_data = pd.read_csv('./embedding/UC1opHUrw8rvnsadT-iGp7Cg/{}.csv'.format(id))
    if cnt == 1:
        print('first:{}-{}'.format(cnt, id))
        cnt += 1
        data = new_data
        end_time = data['offset'].iat[-1]
        # print(new_data['Offset'].to_numpy()[:10])
    else:
        print('next:{}-{}'.format(cnt, id))
        cnt += 1
        new_data['offset'] = new_data['offset'].add(end_time + 3600)
        # print(new_data['Offset'].to_numpy()[:10])
        data = data.append(new_data, ignore_index=True)
        end_time = data['offset'].iat[-1]
        # print(end_time)

print(data.info())
data.to_csv('./concat_week_v3.10_tmp3.csv')


dataset_name = 'concat_week_v3.10_tmp3'
graph_df = pd.read_csv('./{}.csv'.format(dataset_name))
#print(graph_df.info())
#graph_df = pd.read_csv('./dynamicGraph/ml_{}.csv'.format(dataset_name))
#print(graph_df.info())

with open('val.pkl', 'rb') as file:
    val_data = pickle.loads(file.read())
pred_labels = np.load('pred_label.npy')


superchats = graph_df[(graph_df['superchat']>0) & (graph_df['offset']>=val_data.timestamps[0])]
#print(superchats.info(), superchats)
#superchats.to_csv('./{}-superchats.csv'.format(dataset_name))


#pos_idx = np.nonzero(np.logical_and(pred_labels, val_data.labels))
pos_idx = np.nonzero(val_data.labels)
pos_id = []
for idx in pos_idx[0]:
    if val_data.sources[idx] == val_data.destinations[idx]:
        pos_id.append(val_data.sources[idx])
        #print(val_data.sources[idx], val_data.destinations[idx], val_data.edge_idxs[idx], val_data.labels[idx],
        #      pred_labels[idx], val_data.timestamps[idx])
pos_id = np.array(pos_id)  # superchat node id
pos_edge_id = val_data.edge_idxs[pos_idx]

print(pos_id.shape)
print('pos_edge_id:', val_data.edge_idxs[pos_idx])
print('counter pos_id:', Counter(pos_id))
print('counter all_id:', Counter(val_data.sources))


idxs = np.where(val_data.sources == 962  )
surroundings = []
print(len(idxs))
for idx in idxs[0]:
    if val_data.labels[idx] == 1 and pred_labels[idx] == 1 and val_data.timestamps[idx] > 140211: #
        #print(pred_labels[idx], val_data.labels[idx], val_data.edge_idxs[idx])
        print(val_data.sources[idx], val_data.destinations[idx], val_data.edge_idxs[idx], val_data.labels[idx],
              pred_labels[idx], val_data.timestamps[idx])
        #surroundings.append()







print(a)
npy = np.load('embedding/UC1opHUrw8rvnsadT-iGp7Cg/97DWg8tqo4M_aug.npy')
data = pd.read_csv('embedding/UC1opHUrw8rvnsadT-iGp7Cg/97DWg8tqo4M_aug.csv')
print(data.shape)
print(npy.shape)

a = [0, 1, 2]

print(np.tile(a, (2, 1)))

b = np.linspace(0, 9, 20)
print('b1:{}'.format(b))
b = 1 / 10 ** b
print('b2:{}'.format(b))

b = np.tile(b, (2, 1)).T

print('b3:{}'.format(b))

b = np.cos(b)

print('b4:{}'.format(b))




concat_list = ['ON3WijEIS1c', 'qO8Ld-qLjb0', 'k3Nzow_OqQY', 'y3DCfZmX8iA', 'qHZwDxea7fQ', 'cibdBr9TkEo', 'rW8jSXVsW2E', 'eIi8zCPFyng', 'wtJj3CO_YR0', '1kxCz6tt2MU']

for video_id in concat_list:
    #video_id = i #'97DWg8tqo4M'
    print('Video {} augmentation start'.format(video_id))
    old_data = pd.read_csv('embedding/UC1opHUrw8rvnsadT-iGp7Cg/{}.csv'.format(video_id))
    old_emb = np.load('embedding/UC1opHUrw8rvnsadT-iGp7Cg/{}.npy'.format(video_id))
    new_data = old_data.copy(deep=True)
    new_data = new_data[0:0]
    new_emb = np.zeros((1, old_emb.shape[1]))
    #print(new_data)
    for idx, line in old_data.iterrows():
        if idx%1000 == 0:
            print(idx)
    #print(idx, line)
        if int(line['superchat']) > 0:
            for i in range(10):
                new_data = new_data.append(line, ignore_index=True)# = pd.concat([new_data, line], ignore_index=True)
                new_emb = np.append(new_emb, np.expand_dims(old_emb[idx], axis=0), axis=0)
        else:
            new_data = new_data.append(line, ignore_index=True)#new_data = pd.concat([new_data, line], ignore_index=True)
            new_emb = np.append(new_emb, np.expand_dims(old_emb[idx], axis=0), axis=0)

    new_data = new_data.drop(columns=['Unnamed: 0'])
    print('old:', old_data.info())
    print('new:', new_data.info(), new_emb.shape)


    new_data.to_csv('embedding/UC1opHUrw8rvnsadT-iGp7Cg/{}_aug_10.csv'.format(video_id))
    np.save('embedding/UC1opHUrw8rvnsadT-iGp7Cg/{}_aug_10.npy'.format(video_id), new_emb)
    print('Video {} augmentation finished'.format(video_id))



#end_time = new_data['Offset'].values[-1]
#print(end_time, type(end_time))

for (i, j), k in zip(new_data[10:20].iterrows(), range(10)):
  print(i, j, k)


class args():

    def __init__(self):
        self.dim = 0
        self.length = 1

args = args()
print(args.dim)

history_list = list(range(10,0,-1))
delete_list = [2, 3, 5]
print(history_list)
history_list = [history_list[idx] for idx in range(len(history_list)) if idx not in delete_list]
print(history_list)

'''
'''
dataset_name = '1kxCz6tt2MU_v3.10_dynamic_graph'
graph_df = pd.read_pickle('./dynamicGraph/{}.pkl'.format(dataset_name))
#graph_df = pd.read_csv('./dynamicGraph/ml_{}.csv'.format(dataset_name))
print(graph_df.info())
print(graph_df[:10])

'''
'''
print(0>=0.0)
tst = 'なるはや待機� aqua ❤ エペかな、モンハンかな、🥳'
a = '許されたｗ'
b = '許されてるｗ'
c = '許された'
a = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\uAC00-\uD7AF\u3040-\u31FF])","",a)
b = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\uAC00-\uD7AF\u3040-\u31FF])","",b)
c = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\uAC00-\uD7AF\u3040-\u31FF])","",c)

r1 = difflib.SequenceMatcher(None, a, b).real_quick_ratio()
r2 = difflib.SequenceMatcher(None, a, c).real_quick_ratio()

print(a, b, c)
print(r1, r2)
#print(subtst)
'''
#data = pd.read_csv('src/sc_data3.csv')

#data.info()

#print(data[:10])
