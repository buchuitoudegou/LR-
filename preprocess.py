import csv

filename = './data/trainSet.csv'

result = []

with open(filename, 'r') as f:
  csv_reader = csv.reader(f)
  idx = 0
  for row in csv_reader:
    if idx > 0:
      row = list(map(lambda x: float(x), row))
      result.append(row)
    idx += 1
idx = 0
step = len(result) // 5
for i in range(0, len(result), step):
  temp = result[i:i+step]
  with open('./data/fold-{}.csv'.format(idx), 'w', newline='') as f:
    csv_writer = csv.writer(f)
    for row in temp:
      csv_writer.writerow(row)
  idx += 1