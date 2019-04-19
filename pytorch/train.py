import torch.nn as nn
from config import epoches, lr, l2_lambda, lr_decrease_epoch, lr_decrease_ratio
import torch
import matplotlib.pyplot as plt
import numpy as np

def predict(net, testloader, device):
  correct, total = 0, 0
  with torch.no_grad():
    for data in testloader:
      point, label = data['point'], data['label']
      point, label = point.to(device), label.to(device)
      out = net(point)
      predicted = (out > 0.5).type(torch.FloatTensor)
      # print(predicted, label)
      result = predicted == label
      correct += result.sum()
      total += label.size(0)
  return correct.item() / total

def show_running_loss(running_loss, idx):
	x = np.array([i for i in range(len(running_loss))])
	y = np.array(running_loss)
	plt.figure()
	plt.plot(x, y, c='b')
	plt.axis()
	plt.title('loss curve')
	plt.xlabel('step')
	plt.ylabel('loss value')
	plt.savefig('../result/loss-{}.png'.format(idx))

def show_accuracy(running_accuracy, idx):
	x = np.array([i for i in range(len(running_accuracy))])
	y = np.array(running_accuracy)
	plt.figure()
	plt.plot(x, y, c='b')
	plt.axis()
	plt.title('accuracy curve')
	plt.xlabel('step')
	plt.ylabel('accuracy')
	plt.savefig('../result/accuracy-{}.png'.format(idx))

def l2_penalty(var):
  return torch.sqrt(torch.pow(var, 2).sum())

def train(net, trainloader, testloader, device, optim, idxx):
  global lr
  criterion = nn.BCELoss().to(device)
  optimizer = optim
  running_loss = []
  running_accuracy = []
  temp_loss = 0.0
  it = 0
  for epoch in range(epoches):
    if epoch > 0 and epoch % lr_decrease_epoch == 0:
        lr *= lr_decrease_ratio
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr
    for i, data in enumerate(trainloader):
      point, label = data['point'], data['label']
      point, label = point.to(device), label.to(device)
      out = net(point)
      optimizer.zero_grad()
      loss = criterion(out, label)
      loss += l2_lambda * l2_penalty(out)
      loss.backward()
      optimizer.step()
      temp_loss += loss.item()
      it += 1
    running_loss.append(temp_loss / it)
    it = 0
    temp_loss = 0
    if testloader != None:
      running_accuracy.append(predict(net, testloader, device))
      print('train: [%3d/%3d], accuracy: %.4f' % (epoch, epoches, running_accuracy[-1]))
    else:
       print('train: [%3d/%3d], loss: %.4f' % (epoch, epoches, running_loss[-1]))
  if testloader != None:
    show_accuracy(running_accuracy, idxx)
    show_running_loss(running_loss, idxx)
  return net, (running_accuracy[-1] if testloader != None else None)