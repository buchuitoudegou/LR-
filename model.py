import torch.nn as nn
import torch.nn.functional as F
import torch

class LR_Network(nn.Module):
  def __init__(self, n_input, n_output):
    super(LR_Network, self).__init__()
    self.linear = nn.Linear(n_input, n_output)
    # nn.init.xavier_normal(self.linear.weight)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.linear(x)
    x = self.sigmoid(x)
    return x
