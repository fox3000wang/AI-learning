
from torch.utils.data import Dataset

# imageFolder方法


# pytouch处理的格式都是Tensor格式
def transforms():
    from torchvision.datasets.mnist import MNIST
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = MNIST(root="./data", train=True,
                    download=True, transform=transform)


# 自定义数据集
def mnist():
    from torchvision.datasets.mnist import MNIST
    import matplotlib.pyplot as plt
    dataset = MNIST(root="./data", train=True, download=True, transform=None)
    plt.imshow(dataset[0][0], cmap="gray")


# dataset的定义
def MyDataSet():
  # 重写Dataset类
    class MyDataset(Dataset):
        def __init__(self):
            pass

        def __len__(self):
            return 100

        def __getitem__(self, index):
            return index

    # python 提供的magic method
    # p.__getitem__(i) 和 p[i] 等价
    # p.__len__() 和 len(p) 等价


if __name__ == "__main__":

    # mnist()
