import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Train_Dataset(Dataset):
    def __init__(self):
        data = np.loadtxt('E:/PSG/data_label/partial2.txt', delimiter=',', dtype=np.float32)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, :-1])
        self.y_data = torch.LongTensor(data[:, -1])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = Train_Dataset()
batch_size = 11
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
data_test = np.loadtxt('E:/PSG/data_label/partial_test.txt', delimiter=',', dtype=np.float32)
x_test = torch.from_numpy(data_test[:batch_size, :-1])
y_test = torch.LongTensor(data_test[:batch_size, -1])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=50, stride=6, padding=68)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=8)
        self.fc = torch.nn.Linear(64, 5)

    def forward(self, x):
        x = x.view(-1, 1, 3000)
        x = F.relu(self.maxpool(self.cnn(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


net = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.04)


def train(num_epochs):
    for epoch in range(num_epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print('Epoch: %d, Loss: %.3f' % (epoch + 1, loss.item()))
    _, prediction = outputs.max(dim=1)
    print('='*5, '训练集上的预测', '='*5)
    print('Labels: ', labels)
    print('Predict: ', prediction)

    _, prediction = outputs.max(dim=1)
    print('='*5, '测试集上的预测', '='*5)
    outputs = net(x_test)
    _, prediction = outputs.max(dim=1)
    print('Labels: ', y_test)
    print('Predict: ', prediction)


train(30)
