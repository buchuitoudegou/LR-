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
  trainset = MyDataset(['./data/fold-0.csv', './data/fold-1.csv', './data/fold-2.csv','./data/fold-3.csv'], transform=transform)
  testset = MyDataset(['./data/fold-4.csv'], transform=transform)
  trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
  testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
  net = LR_Network(32, 1)
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
  net = train(net, trainloader, testloader, device, optimizer)