# -*- coding: utf-8 -*-
# maxpool 最大池的作用是: 保留输入的特征, 把数据量减小, 减少运算量, 减少内存占用, 减少计算时间
# 可以看成把1080p的视频转成720p。

import os
import torch
import torchvision  # For datasets

from torch import nn  # Neural Network Module
from torch.nn import MaxPool2d  # MaxPooling
from torch.utils.data import DataLoader  # For loading data
from torchvision.transforms import ToTensor

from torch.utils.tensorboard import SummaryWriter


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.maxPool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        return self.maxPool(input)

# 清空logs文件夹


def cleanLogs():
    if os.path.exists('logs'):
        for root, dirs, files in os.walk('logs', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))


def main():
    cleanLogs()
    t = MyModule()

    dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    dataloader = DataLoader(dataset, batch_size=64)

    print(len(dataset))  # 训练:50000 测试:10000

    i = 1
    writer = SummaryWriter("logs")  # 数据放log目录
    for data in dataloader:
        imgs, targets = data
        print(imgs.shape)
        writer.add_images("input_set", imgs, i)

        output = t(imgs)
        print(output.shape)
        writer.add_images("output_set", output, i)

        i += 1

    writer.close()


if __name__ == '__main__':
    main()
