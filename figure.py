import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np


import plotly.io as pio
pio.renderers.default = "browser"



fig = make_subplots(
    rows=4, cols=3)

lambda_x = [str(x) for x in [0.01, 0.05, 0.1, 0.5, 1.0]]
alpha_x = [str(x) for x in [0.01, 0.05, 0.1, 0.5, 0.9]]
dim_x = [str(x) for x in [16, 32, 64, 128, 256]]



col = ['YouTube', 'Mooc', 'Reddit', 'Wikipedia']



lambda_youtube = [0.5076, 0.5074, 0.5076, 0.5076, 0.5076]
lambda_mooc = [0.0925, 0.0889, 0.0878, 0.0903, 0.0903]
lambda_reddit = [0.0376, 0.0369, 0.0381, 0.0368, 0.0512]
lambda_wiki = [0.0272, 0.0274, 0.0276, 0.0271, 0.0312]

df_lambda = pd.DataFrame(np.array([lambda_youtube, lambda_mooc, lambda_reddit, lambda_wiki]).T,
                         columns=col)


alpha_youtube = [0.5074, 0.5078, 0.5076, 0.5077, 0.5076]
alpha_mooc = [0.0895, 0.0913, 0.0903, 0.0932, 0.0925]
alpha_reddit = [0.0374, 0.0368, 0.0368, 0.0373, 0.0371]
alpha_wiki = [0.0270, 0.0274, 0.0272, 0.0274, 0.0275]


df_alpha = pd.DataFrame(np.array([alpha_youtube, alpha_mooc, alpha_reddit, alpha_wiki]).T,
                         columns=col)


dim_youtube = [0.5080, 0.5081, 0.5076, 0.5076, 0.5076]
dim_mooc = [0.0911, 0.0893, 0.0806, 0.0804, 0.0804]
dim_reddit = [0.0373, 0.0368, 0.0368, 0.0363, 0.0368]
dim_wiki = [0.0272, 0.0276, 0.0277, 0.0271, 0.0275]


df_dim = pd.DataFrame(np.array([dim_youtube, dim_mooc, dim_reddit, dim_wiki]).T,
                         columns=col)

idx = [1, 2, 3, 4]
for i in range(len(col)):
    c = idx[i]
    col_name = col[i]
    fig.add_trace(go.Scatter(x=alpha_x, y=df_alpha[col_name],
                   mode='lines',
                   name=col_name), row=c, col=1)

    fig.add_trace(go.Scatter(x=lambda_x, y=df_lambda[col_name],
                             mode='lines',
                             name=col_name), row=c, col=2)

    fig.add_trace(go.Scatter(x=dim_x, y=df_dim[col_name],
                             mode='lines',
                             name=col_name), row=c, col=3)

#fig1.update_layout(title='Model performance with different value of α', xaxis_title='value of α', yaxis_title='RMSE score')

#fig2 = go.Scatter(x=df_lambda['name'], y=df_lambda[y_axis])
#fig2.update_layout(title='Model performance with different value of λ', xaxis_title='value of λ', yaxis_title='RMSE score')


#fig3 = go.Scatter(x=df_dim['name'], y=df_dim[y_axis])
#fig3.update_layout(title='Model performance with different hidden embedding dimensions',
#                  xaxis_title='Dimension of hidden embeddings', yaxis_title='RMSE score')


#fig.add_trace(fig1, row=1, col=1)
#fig.add_trace(fig2, row=1, col=2)
#fig.add_trace(fig3, row=1, col=3)

fig.update_layout(height=2400, width=3600, showlegend=False, font=dict(size=36))
fig.update_traces(line={'width': 10})


fig.update_yaxes(title_text="YouTube", row=1, col=1)
fig.update_yaxes(title_text="Mooc", row=2, col=1)
fig.update_yaxes(title_text="Reddit", row=3, col=1)
fig.update_yaxes(title_text="Wikipedia", row=4, col=1)

fig.update_xaxes(title_text="value of α", row=4, col=1)
fig.update_xaxes(title_text="value of λ", row=4, col=2)
fig.update_xaxes(title_text="hidden dimension", row=4, col=3)

fig.for_each_xaxis(lambda axis: axis.title.update(font=dict(size=48)))
fig.for_each_yaxis(lambda axis: axis.title.update(font=dict(size=48)))




# Update yaxis properties
fig.update_yaxes(dtick=0.0002, row=1, col=1)
fig.update_yaxes(dtick=0.0002, row=1, col=2)
fig.update_yaxes(dtick=0.0002, row=1, col=3)
fig.update_yaxes(dtick=0.002, row=2, col=1)
fig.update_yaxes(dtick=0.002, row=2, col=2)
fig.update_yaxes(dtick=0.005, row=2, col=3)
fig.update_yaxes(dtick=0.0002, row=3, col=1)
fig.update_yaxes(dtick=0.01, row=3, col=2)
fig.update_yaxes(dtick=0.0004, row=3, col=3)
fig.update_yaxes(dtick=0.0002, row=4, col=1)
fig.update_yaxes(dtick=0.002, row=4, col=2)
fig.update_yaxes(dtick=0.0002, row=4, col=3)

fig.update_xaxes(showgrid=True, gridwidth=7)
fig.update_yaxes(showgrid=True, gridwidth=7)


#fig['layout']['yaxis1'].update(domain=[0, 0.2])
#fig['layout']['yaxis2'].update(domain=[0.3, 0.7])
#fig['layout']['yaxis3'].update(domain=[0.8, 1])

fig.show()

fig.write_image('parameter.png')
