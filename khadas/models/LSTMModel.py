import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class SIBILSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length):
    super(SIBILSTMModel, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.seq_length = seq_length

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True)
    self.flatten = nn.Flatten()
    self.fc = nn.Linear(in_features=seq_length*hidden_size, out_features=hidden_size)
    self.bn = nn.BatchNorm1d(hidden_size)

    # Final output
    self.output = nn.Linear(hidden_size, output_size)
    self.eval()

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

    out, _ = self.lstm(x, (h0, c0))
    out_flat = self.flatten(out)

    out_fc = self.fc(out_flat)
    out_bn = self.bn(out_fc)
    
    final_out = self.output(F.relu(out_bn))
    return final_out