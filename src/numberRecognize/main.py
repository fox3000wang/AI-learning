
'''
手写数字识别
'''
import datetime

import torch
from torch import nn  # 神经网络模块
from torch.nn import functional as F  # 激励函数
from torch import optim  # 优化工具包

import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

start = datetime.datetime.now()

# 定义超参数
BATCH_SIZE = 16  # 批大小
# BATCH_SIZE = 512  # 批大小
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备
# EPOCHS = 10  # 训练次数
EPOCHS = 1  # 训练次数

print('batch_size:{} use:{} epochs:{}'.format(BATCH_SIZE, DEVICE, EPOCHS))


# 构建pipline
pipline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))  # 正则化，降低模型复杂度
])


# 下载数据集
train_set = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=pipline
)

test_set = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=pipline
)

print('train_set:{} test_set:{}'.format(len(train_set), len(test_set)))

# 加载数据集
train_loder = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)


# 构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 1.灰度通道 10.输出通道 5.卷积核大小
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)  # 1.输入通道 20.输出通道 3.卷积核大小
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 1.输入维度 50.输出维度
        self.fc2 = nn.Linear(500, 10)  # 1.输入维度 50.输出维度

    def forward(self, x):
        input_size = x.size(0)  # batch_size
        x = self.conv1(x)
        x = F.relu(x)  # 激励函数
        x = F.max_pool2d(x, 2, 2)  # 池化
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(input_size, -1)  # 展平
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)  # 计算分类后，每个数字的概率
        return output


# 定义优化器
model = Digit().to(DEVICE)
# optimizer = optim.Adam(module.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters())


# 定义训练方法
def train_model(model, device, train_loder, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loder):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.max(dim=1, keepdim=True)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loder.dataset),
                100. * batch_idx / len(train_loder), loss.item()))


# 定义测试方法
def test_model(model, device, test_loader):
    model.eval()
    correct = 0.0
    rest = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = F.cross_entropy(output, target).item()
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss = test_loss / len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


# 调用
for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loder, optimizer, epoch)
    test_model(model, DEVICE, test_loader)


end = datetime.datetime.now()
print('total time:', (end - start).seconds)
