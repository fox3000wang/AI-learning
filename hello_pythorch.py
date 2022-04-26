from PIL import Image
import os

import torch  # 这一行没有报错说明环境已经
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.tensorboard import SummaryWriter   # 数据可视化

import numpy as np


# 查看torch模块的所有方法
# dir(torch)

# 查看torch.nn.Module的帮助文档
# help(torch.nn.Module)


# 数据集类
class MyData(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.path = os.path.join(data, label)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.path, self.label, img_name)
        img = Image.open(img_item_path)
        label = self.label
        return img, label

    def __len__(self):
        return len(self.img_path)


def main():

    # 查看显卡是否支持
    print("torch.cuda.is_available():", torch.cuda.is_available())

    # 数据集
    root_path = "./data/train"
    ants_label_dir = "ants_image"
    bees_label_dir = "bees_image"
    ants_dataset = MyData(root_path, ants_label_dir)
    bees_dataset = MyData(root_path, bees_label_dir)

    # 数据可视化工具 TensorBoard
    writer = SummaryWriter("logs")  # 数据放log目录

    image_path = "data/train/ants_image/6240329_72c01e663e.jpg"
    img_PIL = Image.open(image_path)  # 打开图片
    img_array = np.array(img_PIL)    # 将PIL图片转换为numpy数组
    print(type(img_array))
    print(img_array.shape)

    # 图片数据写入
    writer.add_image("train", img_array, 1, dataformats='HWC')  # 数据格式为HWC

    for i in range(100):  # 数据可视化
        writer.add_scalar("y=2x", 3*i, i)  # 数据格式为scalar

    writer.close()

    # 训练数据加载器
    training_set = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),  # 将PIL图片转换为Tensor
    )

    # print(training_set[0])
    for i in range(20):
        img, target = training_set[i]
        writer.add_image("test_set", img, i)


# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )

# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

# batch_size = 64

# # Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break


if __name__ == '__main__':
    main()

'''
张量（tensor）理论是数学的一个分支学科，在力学中有重要应用。张量这一术语起源于力学，
它最初是用来表示弹性介质中各点应力状态的，后来张量理论发展成为力学和物理学的一个有力的数学工具。
张量之所以重要，在于它可以满足一切物理定律必须与坐标系的选择无关的特性。张量概念是矢量概念的推广，
矢量是一阶张量。张量是一个可用来表示在一些矢量、标量和其他张量之间的线性关系的多线性函数。
'''
