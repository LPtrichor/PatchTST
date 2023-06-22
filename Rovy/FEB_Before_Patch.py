import torch
import torch.nn as nn
from .FEBModule import FEBModule_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def FBP(x):
    # x = torch.randn(32, 96, 7)
    # 用一个线性层映射 x形状变为 (32, 96, 512)
    x = x.to(device)
    linear = nn.Linear(7, 512).to(device)
    x = linear(x)
    x = FEBModule_test(x)
    re_linear = nn.Linear(512, 7).to(device)
    x = re_linear(x)
    # print(x.shape)
    return x

# x = torch.randn(32, 96, 7)
# # 用一个线性层映射 x形状变为 (32, 96, 512)
# linear = nn.Linear(7, 512)
# x = linear(x)
# x = FEBModule_test(x)
# re_linear = nn.Linear(512, 7)
# x = re_linear(x)
# print(x.shape)


