import torch
from torch.utils.data import DataLoader
from dataset import MyDataset
from config import batch_size, lr, momentum
import torch.optim as optim
from model import NN
from train import train
import csv
import numpy as np
from sklearn import preprocessing

cuda_available = torch.cuda.is_available()

def predict():
  result = []
  idx = 0
  with open('../data/test set.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
      if idx > 0:
        row = list(map(lambda x: float(x), row))
        result.append(row)
      idx += 1
  result = np.array(result)
  result = preprocessing.scale(result)
  model_dict = torch.load('../result/NN.pkl')
  model = NN(32, 1)
  model.load_state_dict(model_dict)
  model.eval()
  predicted = model(torch.tensor(result).type(torch.FloatTensor))
  predicted = predicted > 0.5
  predicted = predicted.detach().numpy()
  with open('../result/result.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'predicted'])
    for i in range(predicted.shape[0]):
      csv_writer.writerow([i + 1, predicted[i, 0]])

def train_model():
  device = torch.device('cpu')
  transform = [torch.tensor]
  train_files = ['../data/fold-{}.csv'.format(i) for i in range(5)]
  print(train_files)
  trainset = MyDataset(train_files, transform=transform)
  trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
  net = NN(32, 1)
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
  net, _ = train(net, trainloader, None, device, optimizer, 0)
  torch.save(net.state_dict(), '../result/NN.pkl')

def train_test():
  device = torch.device('cpu')
  transform = [torch.tensor]
  accuracies = []
  for idx in range(5):
    train_files = ['../data/fold-{}.csv'.format(i) for i in range(5)]
    train_files = list(filter(lambda x: x != '../data/fold-{}.csv'.format(idx), train_files))
    test_files = ['../data/fold-{}.csv'.format(idx)]
    print(train_files, test_files)
    trainset = MyDataset(train_files, transform=transform)
    testset = MyDataset(test_files, transform=transform)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    net = NN(32, 1)
    if cuda_available:
      net = net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net, acc = train(net, trainloader, testloader, cuda_available, optimizer, idx)
    accuracies.append(acc)
  print(sum(accuracies) / len(accuracies))

if __name__ == "__main__":
  train_test()
  # train_model()
  # predict()