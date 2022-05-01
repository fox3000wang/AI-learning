import datetime
import torch
import torchvision  # For datasets
from torch import nn  # Neural Network Module
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader  # For loading data
from torchvision.transforms import ToTensor


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


start = datetime.datetime.now()

dataset = torchvision.datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)
dataloader = DataLoader(dataset, batch_size=4)

print(len(dataset))  # 训练:50000 测试:10000
print(len(dataloader))  # 训练:782 测试:157

t = Tudui()

i = 1
for data in dataloader:
    imgs, targets = data
    #print(i, imgs.shape)
    output = t(imgs)
    i += 1

end = datetime.datetime.now()

print((end - start).seconds)
