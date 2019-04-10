import torch.nn as nn
from config import epoches, lr
import torch
import matplotlib.pyplot as plt

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

def show_running_loss(running_loss):
	x = np.array([i for i in range(len(running_loss))])
	y = np.array(running_loss)
	plt.figure()
	plt.plot(x, y, c='b')
	plt.axis()
	plt.title('loss curve')
	plt.xlabel('step')
	plt.ylabel('loss value')
	plt.show()

def show_accuracy(running_accuracy):
	x = np.array([i for i in range(len(running_accuracy))])
	y = np.array(running_accuracy)
	plt.figure()
	plt.plot(x, y, c='b')
	plt.axis()
	plt.title('accuracy curve')
	plt.xlabel('step')
	plt.ylabel('accuracy')
	plt.show()

def train(net, trainloader, testloader, device, optim):
  criterion = nn.BCELoss().to(device)
  optimizer = optim
  running_loss = []
  running_accuracy = []
  temp_loss = 0.0
  it = 0
  for epoch in range(epoches):
    for i, data in enumerate(trainloader):
      point, label = data['point'], data['label']
      point, label = point.to(device), label.to(device)
      out = net(point)
      optimizer.zero_grad()
      loss = criterion(out, label)
      loss.backward()
      optimizer.step()
      temp_loss += loss.item()
      it += 1
      # print(it)
    running_loss.append(temp_loss / it)
    it = 0
    temp_loss = 0
    running_accuracy.append(predict(net, testloader, device))
    print('train: [%3d/%3d], accuracy: %.4f' % (epoch, epoches, running_accuracy[-1]))
  show_accuracy(running_accuracy)
  show_running_loss(running_loss)