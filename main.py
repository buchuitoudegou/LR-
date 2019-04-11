import torch
from torch.utils.data import DataLoader
from dataset import MyDataset
from config import batch_size, lr, momentum
import torch.optim as optim
from model import LR_Network
from train import train

if __name__ == "__main__":
  device = torch.device('cpu')
  transform = [torch.tensor]
  accuracies = []
  for idx in range(5):
    train_files = ['./data/fold-{}.csv'.format(i) for i in range(5)]
    train_files = list(filter(lambda x: x != './data/fold-{}.csv'.format(idx), train_files))
    test_files = ['./data/fold-{}.csv'.format(idx)]
    print(train_files, test_files)
    trainset = MyDataset(train_files, transform=transform)
    testset = MyDataset(test_files, transform=transform)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    net = LR_Network(32, 1)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net, acc = train(net, trainloader, testloader, device, optimizer, idx)
    accuracies.append(acc)
  print(sum(accuracies) / len(accuracies))