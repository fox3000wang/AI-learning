# 人工智能学习笔记(入门到放弃)

### 主要分为如下几个部分：

- **数学基础**：包括微积分、线性代数、概率论等对理解机器学习算法有帮助的基本数学。
- **Python**：Python 提供了非常丰富的工具包，非常适合学习者实现算法，也可以作为工业环境完成项目。主流的深度学习框架，例如当前最流行的两个 AI 框架 TensorFlow、PyTorch 都以 Python 作为首选语言。此外，主流的在线课程（比如 Andrew Ng 在 Coursera 的深度学习系列课程）用 Python 作为练习项目的语言。在这部分，我将介绍包括 Python 语言基础和机器学习常用的几个 Library，包括 Numpy、Pandas、matplotlib、Scikit-Learn 等。
- **机器学习**：介绍主流的机器学习算法，比如线性回归、逻辑回归、神经网络、SVM、PCA、聚类算法等等。
- **深度学习**：介绍原理和常见的模型（比如 CNN、RNN、LSTM、GAN 等）和深度学习的框架（TensorFlow、Keras、PyTorch）。
- **强化学习**：介绍强化学习的简单原理和实例。
- **实践项目**：这里将结合几个实际的项目来做比较完整的讲解。此外结合 Kaggle、阿里云天池比赛来做讲解。
- **阅读论文**：如果你追求更高和更深入的研究时，看深度学习各细分领域的论文是非常必要的。

### 重磅 | 完备的 AI 学习路线，最详细的资源整理！

https://zhuanlan.zhihu.com/p/64052743

### 书

《利用 python 进行数据分析》

代码

https://github.com/wesm/pydata-book

公开课

吴恩达《Machine Learning》

作业
https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes

## 相关工具安装

### 安装 octave (可选)

octave 官网
https://www.gnu.org/software/octave/index#

```shell
brew update
brew install octave
```

启动

```shell
# do shell script "/usr/local/bin/octave --gui"  太复杂！

octave --gui
```

### 安装 jupyter

```shell
brew install jupyterlab

jupyter notebook
```

慕课网
liuyubobobo 老师的课

### 安装 pyTorch

先安装 anaconda

https://www.anaconda.com/

再安装依赖

https://pytorch.org/get-started/locally/#macos-version

速度慢就需要切换源

### 安装其他依赖

```shell

pip3 install opencv-python

pip3 install image

pip3 install tensorboard

```

### Copilot 快捷键

```shell

Copilot 也提供了一些快捷键，可以很方便地使用。

接受建议：Tab
拒绝建议：Esc
打开 Copilot：Ctrl + Enter （会打开一个单独的面板，展示 10 个建议）
下一条建议：Alt/Option + ]
上一条建议：Alt/Option + [
触发行内 Copilot：Alt/Option + \ （Coplit 还没有给出建议或者建议被拒绝了，希望手工触发它提供建议）

```
