from PIL import Image
import os

import torch  # 这一行没有报错说明环境已经
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.tensorboard import SummaryWriter   # 数据可视化

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

    root_path = "./data/train"
    ants_label_dir = "ants_image"
    bees_label_dir = "bees_image"
    ants_dataset = MyData(root_path, ants_label_dir)
    bees_dataset = MyData(root_path, bees_label_dir)

    writer = SummaryWriter("logs")

    # writer.add_image("train", img_array, 1, dataformats='HWC')
    for i in range(100):
        writer.add_scalar("y=2x", 3*i, i)

    writer.close()

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
