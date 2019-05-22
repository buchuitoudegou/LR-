from torch.utils.data import Dataset
from loader import read_csv
import numpy as np
import torch

class MyDataset(Dataset):
  """
  Args:
  - files: array of filename
  - transform: transform function
  """
  def __init__(self, files, transform):
    data_point, data_label = None, None
    for filename in files:
      temp_point, temp_label = read_csv(filename)
      if data_point is None:
        data_point, data_label = temp_point, temp_label
      else:
        data_point = np.vstack((data_point, temp_point))
        data_label = np.vstack((data_label, temp_label))
    self.frame = [data_point, data_label]
    self.transform = transform
  
  def __len__(self):
    return len(self.frame[0])
  
  def __getitem__(self, idx):
    point = self.frame[0][idx]
    label = self.frame[1][idx]
    if self.transform != None:
      for fn in self.transform:
        point = fn(point)
        label = fn(label)
    sample = { 'point': point.type(torch.FloatTensor), 'label': label.type(torch.FloatTensor) }
    return sample