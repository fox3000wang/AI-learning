# -*- coding: utf-8 -*-
import torch
import torchvision  # For datasets
from torch import nn  # Neural Network Module
from torch.nn import Conv2d  # Convolutional Layer
from torch.utils.data import DataLoader  # For loading data
from torchvision.transforms import ToTensor

from torch.utils.tensorboard import SummaryWriter   # 数据可视化

# Define the neural network module


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(3, 6, 3)  # 卷积层

    def forward(self, input):
        return self.conv1(input)


def main():
    dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    dataloader = DataLoader(dataset, batch_size=64)

    print(len(dataset))  # 训练:50000 测试:10000
    print(len(dataloader))  # 训练:782 测试:157

    t = Tudui()
    writer = SummaryWriter("logs")  # 数据放log目录

    i = 1
    for data in dataloader:
        imgs, targets = data
        print(i, imgs.shape)
        writer.add_images("base_set", imgs, i)
        output = torch.reshape(t(imgs), (-1, 3, 30, 30))
        writer.add_images("output_set", output, i)
        i += 1

    writer.close()


if __name__ == '__main__':
    main()
