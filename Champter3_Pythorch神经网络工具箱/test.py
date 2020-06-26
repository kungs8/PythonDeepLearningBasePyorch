# -*- encoding: utf-8 -*-
'''
@Time    :   2020/6/26:下午6:01
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
'''

import cmath

num = int(input("请输入一个数字: "))
num_sqrt = cmath.sqrt(num)
print('{0} 的平方根为 {1:0.3f}+{2:0.3f}j'.format(num ,num_sqrt.real,num_sqrt.imag))
print(num_sqrt ** num_sqrt)