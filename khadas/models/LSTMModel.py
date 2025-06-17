import torch
import torch.nn as nn
import torch.nn.functional as Functional

torch.manual_seed(0)

class SIBILSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length):
    super(SIBILSTMModel, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.seq_length = seq_length

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True, bidirectional=True)
    self.flatten = nn.Flatten()
    self.bn = nn.BatchNorm1d(seq_length*hidden_size*2)
    self.fc1 = nn.Linear(in_features=seq_length*hidden_size*2, out_features=hidden_size)
    self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)


  def forward(self, x):
    num_directions = 2
    h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size)
    c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size)

    out, _ = self.lstm(x, (h0, c0))
    out_flat = self.flatten(out)

    out_fc1 = self.fc1(self.bn(Functional.relu(out_flat)))
    out_fc2 = self.fc2(Functional.relu(out_fc1))
    pred = out_fc2

    return pred