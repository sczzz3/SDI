import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import json

x = []
with open('/Users/aoao/Desktop/x', 'r') as fx:
    for i in fx.readlines():
        tmp = json.loads(i)
        x.append(tmp)
# print(len(x[2649]))  ?????
x[2649].append(0)
x = np.array(x)

y = []
with open('/Users/aoao/Desktop/y', 'r') as fy:
    for j in fy.readlines():
        tmp = str(j[0])
        y.append(tmp)

for k in range(len(y)):
    if y[k] == 'W':
        y[k] = 0
    if y[k] == '1':
        y[k] = 1
    if y[k] == '2':
        y[k] = 2
    if y[k] == '3' or y[k] == '4':
        y[k] = 3
    if y[k] == 'R':
        y[k] = 4
y = np.array(y)


class EEGDataset(Dataset):
    def __init__(self):
        self.len = len(y)
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = EEGDataset()
train_loader = DataLoader(dataset=dataset, batch_size=25, shuffle=False)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(3000, 500)
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(1, 1))
        self.pooling = torch.nn.MaxPool2d(1)
        self.lstm = torch.nn.LSTM(input_size=500, hidden_size=100, num_layers=2)
        self.fc = torch.nn.Linear(100, 5)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = x.view(25, 1, 1, 500)
        x = F.relu(self.pooling(self.conv1(x)))
        x = x.view(1, 25, 500)
        hidden1 = torch.zeros(2, 25, 100)
        hidden2 = torch.zeros(2, 25, 100)
        x, _ = self.lstm(x, (hidden1, hidden2))
        x = x.view(25, -1)
        x = F.relu((self.fc(x)))
        return x


model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs = inputs.to(torch.float32)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print('[%d, %3d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss))
        running_loss = 0.0


for i in range(3):
    train(i)


