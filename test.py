dicts = {}

labels = [-1, 2, -1, 0, 1, 1, 0, 2, 0]
from itertools import groupby

for k, g in groupby(sorted(labels, reverse=True)):
    dicts[k] = len(list(g))
print(dicts.keys())
print(dicts.values())
a = list(dicts.keys())+ list(dicts.values())
print("a:", a)

import torch
from torch import nn
m = nn.Sigmoid()
input = torch.randn(2)
output = m(input)
print(output)