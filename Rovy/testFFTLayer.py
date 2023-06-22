import torch
from FFTLayer import FourierBlock

# 创建一个FourierBlock模块
block = FourierBlock(in_channels=512, out_channels=512, seq_len=144, modes=64)

# 创建输入张量
# 在PyCharm中，向上复制一行的快捷键是Ctrl + D。你可以将光标放在要复制的行上，然后按下Ctrl + D，即可复制该行并将其插入到上面的一行
# x = torch.randn(16,96,8,64) # 可以运行的
# x = torch.randn(224,12,512) # 测试
# 这里的512分为8个head
x = torch.rand(224, 12, 8, 64)

# x = torch.randn(32,7,12,512)
# x = torch.randn(3)
# 调用模块的前向传播方法
y, _ = block(x, x, x, None)

# 打印输出张量的形状
print(y.shape)

# # 用pytorch生成形状为(32, 96, 7)的tensor
# import torch
# from FFTLayer import FourierBlock
#
# x = torch.randn(32, 96, 7)
#
# fft_block = FourierBlock(3, 3, 96, modes=64, mode_select_method='random')
# x_fft = FourierBlock(x, x, x)
#
# print(x_fft)
#
#
