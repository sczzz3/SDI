import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import json

class Train_Dataset(Dataset):
    def __init__(self, datafile_train, labels_train):
        datax = np.loadtxt(datafile_train, delimiter=',', dtype=str)
        datax = datax[:-1]
        datax = datax.reshape(-1, 3000)
        datax = datax.astype(np.float32)
        self.x_data = torch.from_numpy(datax)
        self.len = self.x_data.shape[0]
        datay = np.loadtxt(labels_train, dtype=np.float32)
        self.y_data =torch.LongTensor(datay)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Test_Dataset(Dataset):
    def __init__(self, datafile_test, labels_test):
        testx = np.loadtxt(datafile_test, delimiter=',', dtype=str)
        testx = testx[:-1]
        testx = testx.reshape(-1, 3000)
        testx = testx[:976]
        testx = testx.astype(np.float32)
        self.x_test = torch.from_numpy(testx)
        self.len = self.x_test.shape[0]
        testy = np.loadtxt(labels_test, dtype=np.float32)
        self.y_test =torch.LongTensor(testy)

    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(3000, 500)
        self.conv1 = torch.nn.Conv1d(1, 200, kernel_size=1)
        self.pooling = torch.nn.MaxPool1d(1)
        self.lstm = torch.nn.LSTM(input_size=500, hidden_size=100, num_layers=1)
        self.fc1 = torch.nn.Linear(20000, 3000)
        self.fc2 = torch.nn.Linear(200, 20)
        self.fc3 = torch.nn.Linear(20, 5)
        self.bn1 = torch.nn.BatchNorm1d(200)
        self.bn2 = torch.nn.BatchNorm1d(2)
        self.dp = torch.nn.Dropout(0.3)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        x = x.view(batch_size, 1, 500)

        x = F.relu(self.pooling(self.conv1(x)))
        x = self.bn1(x)
        # x = self.dp(x)

        x = x.view(-1, batch_size, 500)
        hidden1 = torch.zeros(1, batch_size, 100)
        hidden2 = torch.zeros(1, batch_size, 100)
        _, x = self.lstm(x, (hidden1, hidden2))

        x = F.tanh(torch.cat([x[-1], x[-2]], dim=1))
        x = self.bn2(x)
        # x = self.dp(x)

        x = x.view(batch_size, -1)
        # x = F.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    # running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs = inputs.to(torch.float32)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # running_loss += loss.item()
        if batch_idx % 100 == 0:
            print('[%d, %3d] loss: %.3f' % (epoch + 1, batch_idx + 1, loss.item()))
        # running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))

# ========================================   Train   =======================================================
# batch_size = 1
# train_dataset = Train_Dataset('E:/data/X_3.txt', 'E:/data/y_3.txt')
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#
# for i in range(20):
#      train(i)
#
#
# batch_size = 3
# train_dataset = Train_Dataset('E:/data/X_2.txt', 'E:/data/y_2.txt')
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#
# for i in range(20):
#      train(i)


batch_size = 1
train_dataset = Train_Dataset('E:/data/X_1.txt', 'E:/data/y_1.txt')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

for i in range(20):
    train(i)


# ==============================================  Test =========================================
batch_size = 1
test_dataset = Test_Dataset('E:/data/X_4.txt', 'E:/data/y_4.txt')
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

test()

