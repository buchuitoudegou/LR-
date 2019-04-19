from sklearn.linear_model import LogisticRegression
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.externals import joblib

def read_csv(filename):
  """
  - params filename: filename of the file
  - params type: string
  - return: numpy ndarray
  """
  with open(filename, mode='r') as f:
    csv_reader = csv.reader(f)
    points = []
    labels = []
    for row in csv_reader:
      row = list(map(lambda x: float(x), row))
      points.append(row[:len(row) - 1])
      labels.append([row[-1]])
    points = np.array(points)
    labels = np.array(labels)
  return points, labels

def load_dataset(files):
  data, labels = None, None
  for filename in files:
    temp_point, temp_label = read_csv(filename)
    if data is None:
      data, labels = temp_point, temp_label
    else:
      data = np.vstack((data, temp_point))
      labels = np.vstack((labels, temp_label))
  return data, labels


def train_test():
  all_acc = []
  for idx in range(5):
    train_files = ['../data/fold-{}.csv'.format(i) for i in range(5)]
    train_files = list(filter(lambda x: x != '../data/fold-{}.csv'.format(idx), train_files))
    test_files = ['../data/fold-{}.csv'.format(idx)]
    train_data, train_labels = load_dataset(train_files)
    test_data, test_labels = load_dataset(test_files)
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(train_data, train_labels)
    result = clf.predict(test_data)
    result = result.reshape((-1, 1))
    acc = result == test_labels
    acc = acc.sum() / test_labels.shape[0]
    all_acc.append(acc)
  print(sum(all_acc) / len(all_acc))

def train_model():
  train_files = ['../data/fold-{}.csv'.format(i) for i in range(5)]
  train_data, train_labels = load_dataset(train_files)
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
  clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(train_data, train_labels)
  predicted = clf.predict(result)
  predicted = predicted.reshape((-1, 1))
  with open('../result/result.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'predicted'])
    for i in range(predicted.shape[0]):
      csv_writer.writerow([i + 1, predicted[i, 0]])


if __name__ == "__main__":
  # train_test()
  train_model()