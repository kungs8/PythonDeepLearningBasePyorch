# -*- encoding: utf-8 -*-
'''
@Time    :   2020/7/20:下午3:15
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. torch.nn.MSELoss 均方误差损失函数

torch.manual_seed(10)  # 设置随机种子，为了后续重跑数据一样
loss = nn.MSELoss(reduction="mean")  # reduction 默认为"mean"，reduction可取值none、mean、sum，缺省值为mean
input = torch.randn(1, 2, requires_grad=True)
print("input:", input)

target = torch.randn(1, 2)
print("target:", target)

output = loss(input, target)
print("output:", output)

output.backward()


# 2. torch.nn.CrossEntropyLoss 交叉熵损失，又称对数似然损失、对数损失；二分类时还可称之为逻辑回归损失
# PyTorch中，这里不是严格意义上的交叉熵损失函数，而是先将input经过softmax激活函数，将向量"归一化"成概率形式。然后再与target计算严格意义上的交叉熵损失。
# 在多分类任务中，经常采用softmax激活函数+交叉熵损失函数，因为交叉熵描述了两个概率分布的差异，然而神经网络输出的是向量，并不是概率分布的形式。
# 所以需要softmax激活函数将一个向量进行"归一化"成概率分布的形式，再采用交叉损失函数计算loss
loss = torch.nn.CrossEntropyLoss()

# 假设类别数是5
input = torch.randn(3, 5, requires_grad=True)
# 每个样本对应的类别索引，其值范围为[0, 4]
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()