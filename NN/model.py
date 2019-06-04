import torch.nn as nn
import torch.nn.functional as F
import torch

class NN(nn.Module):
  def __init__(self, n_input, n_output):
    super(NN, self).__init__()
    self.seq = nn.Sequential(
      nn.Linear(n_input, 128),
      nn.ReLU(),
      # nn.Dropout(0.2),
      nn.Linear(128, 512),
      nn.ReLU(),
      nn.Linear(512, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, 128),
      nn.ReLU(),
      nn.Linear(128, 1),
      nn.Sigmoid()
    )
  
  def forward(self, x):
    return self.seq(x)