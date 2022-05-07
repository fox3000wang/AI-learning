# pytorch

[中文官网](https://pytorch.apachecn.org/#/)

文档安装和打开

```sh
npm install -g pytorch-doc-zh
pytorch-doc-zh <port>
# 访问 http://localhost:{port} 查看文档
```

## 重要概念

### 神经网络

神经网络的典型训练过程如下：

- 定义具有一些可学习参数（或权重）的神经网络
- 遍历输入数据集
- 通过网络处理输入
- 计算损失（输出正确的距离有多远）
- 将梯度传播回网络参数
- 通常使用简单的更新规则来更新网络的权重：weight = weight - learning_rate \* gradient

### 损失函数

### tensor 张量

- pytorch 处理的最小单元
- 可以看成一个数组，也可以是一个矩阵
- 它可以做一些运算变成另外一个张量

### autograd 自动差分引擎

### dataset 数据集

dataset 在某些时候是**数据集**，但是在 pytorch 里是数据集的实例。
