import torch.nn as nn
import torch.nn.functional as F
import torch

class NN(nn.Module):
  def __init__(self, n_input, n_output):
    super(NN, self).__init__()
    self.seq = nn.Sequential(
      nn.Linear(n_input, 100),
      nn.ReLU(),
      nn.Linear(100, 50),
      nn.ReLU(),
      nn.Linear(50, 1),
      nn.Sigmoid()
    )
  
  def forward(self, x):
    return self.seq(x)