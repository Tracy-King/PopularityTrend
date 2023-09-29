import numpy as np
import torch



class BlockLSTM(torch.nn.Module):
    def __init__(self, time_steps, num_variables, lstm_hs=64, dropout=0.8, attention=False):
      super().__init__()
      self.lstm = torch.nn.LSTM(input_size=time_steps, hidden_size=lstm_hs, num_layers=num_variables)
      self.dropout = torch.nn.Dropout(p=dropout)
      self.attn = attention

    def attention_net(self, lstm_output, final_state):
      # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
      # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

      batch_size = len(lstm_output)
      # hidden = final_state.view(batch_size,-1,1)
      hidden = final_state[0].unsqueeze(1)
      #print(lstm_output.shape, hidden.shape)
      # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
      attn_weights = torch.mm(lstm_output, hidden).squeeze(0)
      # attn_weights : [batch_size,n_step]
      #print(attn_weights.shape)
      soft_attn_weights = torch.nn.functional.softmax(attn_weights, 1)
      #print(attn_weights.shape, soft_attn_weights.shape)
      # context: [batch_size, n_hidden * num_directions(=2)]
      context = torch.mm(lstm_output.transpose(0, 1), soft_attn_weights)

      return context, soft_attn_weights

    def forward(self, x):
      # input is of the form (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
      #x = torch.transpose(x, 0, 1)
      # lstm layer is of the form (num_variables, batch_size, time_steps)
      #print(x.shape)
      output, (final_hidden_state, final_cell_state) = self.lstm(x)
      #y = torch.transpose(output, 1, 2)
      #print('LSTM', x.shape)
      if self.attn:
        y, _ = self.attention_net(output, final_hidden_state)
      # dropout layer input shape:
      y = self.dropout(output)
      #print(y.shape)
      # output shape is of the form ()

      return y


class BlockFCNConv(torch.nn.Module):
    def __init__(self, in_channel=1, out_channel=64, kernel_size=8, momentum=0.99, epsilon=0.001, squeeze=False):
      super().__init__()
      self.conv = torch.nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
      self.batch_norm = torch.nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
      self.relu = torch.nn.ReLU()

    def forward(self, x):
      # input (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
      #print("BlockFCN", x.shape)
      #print(x.shape)
      x = self.conv(x)
      # input (batch_size, out_channel, L_out)
      x = self.batch_norm(x)
      # same shape as input
      y = self.relu(x)
      return y


class BlockFCN(torch.nn.Module):
  def __init__(self, time_steps, channels=[1, 64, 128, 64], kernels=[5, 3, 2], mom=0.99, eps=0.001):
    super().__init__()
    self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps, squeeze=True)
    self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps, squeeze=True)
    self.conv3 = BlockFCNConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
    output_size = time_steps - sum(kernels) + len(kernels)
    self.global_pooling = torch.nn.AvgPool1d(kernel_size=output_size)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    # apply Global Average Pooling 1D
    y = self.global_pooling(x)
    return y


class LSTMFCN(torch.nn.Module):
    def __init__(self, dim, num_variables=1, attn=False):
      super().__init__()
      self.lstm_block = BlockLSTM(dim, num_variables, attention=attn)
      self.fcn_block = BlockFCN(dim)
      self.linear = torch.nn.Linear(128, 1)

    def forward(self, x):
      # input is (batch_size, time_steps), it has to be (batch_size, 1, time_steps)
      #x = torch.transpose(x, 0, 1)
      #x = x.unsqueeze(1)
      #print(x.shape)
      #x = x.transpose().unsqueeze(1)
      # pass input through LSTM block
      x1 = self.lstm_block(x)
      x = torch.unsqueeze(x, 1)
      # pass input through FCN block
      x2 = self.fcn_block(x).squeeze()
      #print('LSTMFCN', x1.shape, x2.shape)
      # concatenate blocks output
      x = torch.cat([x1, x2], 1)
      # pass through Softmax activation
      y = self.linear(x)

      #print(y.shape)

      return y.squeeze()
