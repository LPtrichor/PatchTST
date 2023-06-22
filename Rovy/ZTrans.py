import torch
import math

# 定义输入信号
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 计算 FFT
fft_x = torch.fft.fft(x)

# 定义复数向量
# z = torch.exp(-2j * torch.tensor([i * 2 * 3.1415 / len(x) for i in range(len(x))]))


# Define the input signal
# x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# Compute the length of the input signal
n = len(fft_x)

# Compute the angles for each point in the Z-transform
angles = torch.tensor([i * 2 * math.pi / n for i in range(n)])

# Compute the complex exponential for each angle
exponents = torch.exp(-2j * angles)

# Construct the complex vector for the Z-transform
z = exponents

# 计算 Z 变换
z_x = fft_x / z

print(z_x)