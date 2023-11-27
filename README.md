# nnTransfer Readme

# Automated Pancreas Segmentation and Fat Fraction Evaluation Basing a Self-Supervised Transfer Learning Network

---

该代码库将提供腹部CT胰腺分割的网络，分割网络的权重并非通过随机初始化，而是通过利用自监督学习从未标签的腹部CT图像学习权重，并最终计算了胰腺脂肪的浸润比。

![Untitled](nnTransfer%20Readme%2074e3153cae1f42f89f297d602ff20717/Untitled.png)

The segmentation and infiltration of fat in one of our cases. In this case, the fat volume was 89.99ml and the fat infiltration percentage was 9.1%

![Untitled](nnTransfer%20Readme%2074e3153cae1f42f89f297d602ff20717/Untitled%201.png)

# Paper

---

# Installation

---

Experimental Setup: The experiment was conducted using Python 3.9 on a Rocky 8.7 system.环境配置文件可参考`environment.yml`

---

# 代码库功能

按照代码库的功能可以将其分为三个部分。

- 自监督学习网络
- 分割网络
- 评价指标和脂肪浸润比计算

### 自监督学习网络：

`infinite_generator_3D.py` ：用于产生腹部CT胰腺周围的立方块，大小为32×64×64

`Self supervised training` : 将`infinite_generator_3D`生成的数据输入分割网络进行训练。

“如果您对这部分自监督网络感兴趣并想应用到自己的数据集中，可参考“[ModelsGenesis/pytorch at master · MrGiovanni/ModelsGenesis (github.com)](https://github.com/MrGiovanni/ModelsGenesis/tree/master/pytorch)””

### 分割网络：

### 评价指标和脂肪浸润比计算：