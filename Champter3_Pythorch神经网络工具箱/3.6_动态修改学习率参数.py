# -*- encoding: utf-8 -*-
'''
@Time    :   2020/6/26:上午10:52
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

修改参数的方式可以通过修改参数opetimizer.params_groups或新建optimizer
新建optimizer比较简单，optimizer十分轻量级，所有开销很小。
新的优化器会初始化动量等状态信息，这对于使用动量的优化器(momentum参数的sgd)可能会造成收敛中的震荡。
这里直接采用修改修改参数optimizer.params_groups
optimizer.param_groups:长度1的list
optimizer.param_groups[0]:长度为6的字典，包括权重参数、lr、momentum等参数
len(optimizer.param_groups[0]) 结果为6

以下是3.2节中动态修改学习率参数代码:
# 动态修改参数学习率
for epoch in range(num_epoches):
    if epoch % 5 ==0:
        optimizer.param_groups[0]["lr"] *= 0.1
        print("new lr value:", optimizer.param_groups[0]["lr"])
    for img, label in train_loader:
        ######


