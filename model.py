import torch.nn as nn

class LR_Network(nn.Module):
  def __init__(self, n_input, n_output):
    super(LR_Network, self).__init__()
    self.output = nn.Sequential(
      nn.Linear(n_input, n_output),
      nn.Sigmoid()
    )
  
  def forward(self, x):
    return self.output(x)
