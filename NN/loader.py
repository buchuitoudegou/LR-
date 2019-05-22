import csv
import numpy as np

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