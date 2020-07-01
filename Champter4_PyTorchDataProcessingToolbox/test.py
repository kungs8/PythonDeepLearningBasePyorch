# -*- encoding: utf-8 -*-
'''
@Time    :   2020/6/29:下午2:14
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''
from PIL import Image
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# X1, y1 = datasets.make_circles(n_samples=5000, factor=.5, noise=.5)
# X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]], random_state=9)
# X = np.concatenate((X1, X2))
# print(X)
# print(X.shape)
# # 绘制数据分布图-----------------------------
# # plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
# # plt.xlabel('petal length')
# # plt.ylabel('petal width')
# # plt.legend(loc=2)
# # plt.show()
# # ---------------------------------------------
# DB = DBSCAN(eps=0.1, min_samples=6).fit(X)
# label_pred = DB.labels_
# print("label:", len(label_pred))
# color = ['pink', 'blue', 'green', 'grey', 'black']
# marker = ['o', '*', '+']
# plt.figure(figsize=(15, 20), facecolor="white")
# for i in range(max(label_pred) + 1):
#     cluster_i = X[label_pred == i]
#     print(len(cluster_i))
#     plt.scatter(cluster_i[:, 0], cluster_i[:, 1], c=color[i % 5], marker=marker[i % 3],
#                 label="class" + str(i + 1) + '-' + str(len(cluster_i)))
# plt.xlabel("XX")
# plt.ylabel("YY")
# # plt.legend(loc=2)
# plt.axis("off")
# plt.show()


x = np.random.randint(low=1, high=100, size=100)
y1 = np.random.randint(low=1, high=100, size=50)
y2 = np.random.randint(low=-100, high=1, size=50)
y = np.append(y1, y2)
np.random.shuffle(x)
np.random.shuffle(y)
x = [0.32243268, 0.22852958, -1.04233205, 1.44809587, 1.08839337, 1.16216385, 1.04826688, 1.20011086, 1.17593194, -1.04233205, -1.04233205, -1.04233205]
x = [0.33243268, 0.34243268, 0.31243268, 0.32343268, -1.04233205, -1.14233205, -1.06233205, -1.12233205, 1.20011086, 1.20011086, 1.04826688]
y = [0.70744544, 0.7936373, 0.7648807, 0.7598229, -0.19871172, -0.17871172, -0.14871172, -0.16871172, 1.13520525, -0.10648887, -0.1648907, -0.11648807]
print(x,
      y)
# data = np.zeros(len(x), len(y))

data = np.asarray([(x[i], y[i]) for i in range(len(x))])
# X1, y1 = datasets.make_circles(n_samples=5, factor=.5, noise=.5)
# X2, y2 = datasets.make_blobs(n_samples=5, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]], random_state=9)
# data = np.concatenate((X1, X2))
print(data)
print(data[:, 0],
      data[:, 1])
# plt.figure(figsize=(5, 10), facecolor="white")
# plt.show()

db = DBSCAN(eps=0.1, min_samples=3).fit(data)
label_pred = db.labels_
print("label_pred:", label_pred)
color = ['blue', 'green', 'grey','pink',  'black']
marker = ['o', '*', '+', '-']
plt.figure(figsize=(10, 10), facecolor="white")
for i in range(max(label_pred) + 1):
    cluster_i = data[label_pred == i]
    print(len(cluster_i))
    # plt.scatter(data[:, 0], data[:, 1], c="pink")
    plt.scatter(cluster_i[:, 0], cluster_i[:, 1], c=color[i % 4], marker=marker[i % 3],
                label="class" + str(i + 1) + '-' + str(len(cluster_i)))
plt.xlabel("XX")
plt.ylabel("YY")
plt.legend(loc=2)
# plt.axis("off")
plt.show()


import tensorflow as tf

# 定义一个简单的计算图，实现向量加法的操作。
input1 = tf.constant([1.0, 2.0, 3.0], name = 'input1')
input2 = tf.Variable(tf.random_uniform([3]), name = 'input2')
output = tf.add_n([input1, input2], name = 'add')

# 生成一个写日志的writer，并将当前的tensorflow计算图写入日志。
# tensorflow提供了多种写日志文件的API
writer = tf.summary.FileWriter('./log', tf.get_default_graph())
writer.close()

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),  # (6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # (16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


dummy_input = torch.rand(13, 1, 28, 28)  # 假设输入13张1*28*28的图片
model = LeNet()
with SummaryWriter(log_dir="logs", comment='LeNet') as w:
    w.add_graph(model, (dummy_input,))
    w.close()