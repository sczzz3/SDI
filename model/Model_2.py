import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import json
from torchvision.models import resnet
from typing import Optional

class Train_Dataset(Dataset):
    def __init__(self):
        datax = np.loadtxt('E:/data/X_1.txt', delimiter=',', dtype=str)
        datax = datax[:-1]
        datax = datax.reshape(-1, 3000)
        datax = datax.astype(np.float32)
        self.x_data = torch.from_numpy(datax)
        self.len = self.x_data.shape[0]
        datay = np.loadtxt('E:/data/y_1.txt', dtype=np.float32)
        self.y_data =torch.LongTensor(datay)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Test_Dataset(Dataset):
    def __init__(self):
        testx = np.loadtxt('E:/data/X_2.txt', delimiter=',', dtype=str)
        testx = testx[:-1]
        testx = testx.reshape(-1, 3000)
        testx = testx[:976]
        testx = testx.astype(np.float32)
        self.x_test = torch.from_numpy(testx)
        self.len = self.x_test.shape[0]
        testy = np.loadtxt('E:/data/y_2.txt', dtype=np.float32)
        self.y_test =torch.LongTensor(testy)

    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index]

    def __len__(self):
        return self.len

train_dataset = Train_Dataset()
test_dataset = Test_Dataset()
batch_size = 1

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
# ================= CNN1 =============================
        self.conv1_1 = torch.nn.Conv1d(in_channels=1,
                                     out_channels=64,
                                     kernel_size=50,
                                     stride=5)
        self.pooling1_1 = torch.nn.MaxPool1d(kernel_size=5)
        self.conv1_2 = torch.nn.Conv1d(in_channels=64,
                                         out_channels=128,
                                         kernel_size=10,
                                         stride=1)
        self.conv1_3 = torch.nn.Conv1d(in_channels=128,
                                         out_channels=128,
                                         kernel_size=10,
                                         stride=1)
        self.conv1_4 = torch.nn.Conv1d(in_channels=128,
                                         out_channels=128,
                                         kernel_size=10,
                                         stride=1)
        self.pooling1_2 = torch.nn.MaxPool1d(kernel_size=7)    # 维度最后为13
# ================== CNN2 ==============================
        self.conv2_1 = torch.nn.Conv1d(in_channels=1,
                                    out_channels=64,
                                    kernel_size=200,
                                    stride=16)
        self.pooling2_1 = torch.nn.MaxPool1d(kernel_size=5)
        self.conv2_2 = torch.nn.Conv1d(in_channels=64,
                                         out_channels=128,
                                         kernel_size=6,
                                         stride=1)
        self.conv2_3 = torch.nn.Conv1d(in_channels=128,
                                         out_channels=128,
                                         kernel_size=6,
                                         stride=1)
        self.conv2_4 = torch.nn.Conv1d(in_channels=128,
                                         out_channels=128,
                                         kernel_size=6,
                                         stride=1)
        self.pooling2_2 = torch.nn.MaxPool1d(kernel_size=2)     # 维度为10
# ================== CNN3 ================================
        self.conv3_1 = torch.nn.Conv1d(in_channels=1,
                                     out_channels=64,
                                     kernel_size=400,
                                     stride=50)
        self.pooling3_1 = torch.nn.MaxPool1d(kernel_size=4)
        self.conv3_2 = torch.nn.Conv1d(in_channels=64,
                                     out_channels=128,
                                     kernel_size=4,
                                     stride=1)
        self.conv3_3 = torch.nn.Conv1d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=4,
                                     stride=1)
        self.conv3_4 = torch.nn.Conv1d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=4,
                                     stride=1)
        self.pooling3_2 = torch.nn.MaxPool1d(kernel_size=2)   # 维度为2
# ================ DP =================================
        self.DP = torch.nn.Dropout(0.5)
# ================ BN =================================
        self.BN1 = torch.nn.BatchNorm1d(64)
        self.BN2 = torch.nn.BatchNorm1d(64)
        self.BN3 = torch.nn.BatchNorm1d(64)
# ================ BiGRU ====================================
        self.gru1 = torch.nn.GRU(input_size=3200,
                                hidden_size=512,
                                num_layers=1,
                                bidirectional=True)
        self.gru2 = torch.nn.GRU(input_size=1024,
                                hidden_size=512,
                                num_layers=1,
                                bidirectional=True)
# ================ FC ======================================
        self.fc1 = torch.nn.Linear(1024,5)
# =======================================================
#         self.downsample = downsample

    def forward(self, x):
        x_add = x
        x = x.view(batch_size, -1, 3000)
        x1 = x
        x2 = x
        x3 = x
# ========= CNN1 =============
        x1 = self.conv1_1(x1)
        x1 = self.pooling1_1(x1)
        x1 = self.DP(x1)
        x1 = self.BN1(x1)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.pooling1_2(x1)
# ========= CNN2 ===============
        x2 = self.conv2_1(x2)
        x2 = self.pooling2_1(x2)
        x2 = self.DP(x2)
        x2 = self.BN2(x2)
        x2 = self.conv2_2(x2)
        x2 = self.conv2_3(x2)
        x2 = self.conv2_4(x2)
        x2 = self.pooling2_2(x2)
# ========== CNN3 ===============
        x3 = self.conv3_1(x3)
        x3 = self.pooling3_1(x3)
        x3 = self.DP(x3)
        x3 = self.BN3(x3)
        x3 = self.conv3_2(x3)
        x3 = self.conv3_3(x3)
        x3 = self.conv3_4(x3)
        x3 = self.pooling3_2(x3)
# ======== Reshape ===============
        x1 = x1.reshape(batch_size, -1)
        x2 = x2.reshape(batch_size, -1)
        x3 = x3.reshape(batch_size, -1)
# ========  Cat ==================
        x = torch.cat((x1, x2, x3), 1)
        x = self.DP(x)
# ======== BiGRU ===================
        x = x.view(1, -1, 3200)
        x, _ = self.gru1(x)
        x = self.DP(x)
        x, _ = self.gru2(x)
        x = self.DP(x)
# ======== add res ====================
#         if self.downsample is not None:
#             x_add = self.downsample(x_add)
#         x_add = x_add[:,:1024]         # 残差连接，防止梯度消失，但是 降采样函数不太会弄
#         x += x_add
#         x = F.relu(x)
# ========  fc  =====================
        x = x.view(-1, 1024)
        x = self.fc1(x)

        return x


model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

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

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
