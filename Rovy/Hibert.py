import torch


def hilbert_transform(x):
    """
    希尔伯特变换
    :param x: 输入信号，形状为 [batch_size, length]
    :return: 希尔伯特变换后的信号，形状为 [batch_size, length]
    """
    # 对信号进行快速傅里叶变换
    x_fft = torch.fft.fft(x)

    # 构造复数向量
    n = x.shape[-1]
    if n % 2 == 0:
        h = torch.zeros(n)
        h[0] = h[-n // 2 + 1] = 1
        h[1:n // 2] = 2
    else:
        h = torch.zeros(n)
        h[0] = 1
        h[1:(n + 1) // 2] = 2

    # 将复数向量转换为张量
    h = h.to(x.device)
    h = torch.stack([h, torch.zeros_like(h)], dim=-1)

    # 将傅里叶变换后的信号与复数向量进行点乘
    y_fft = x_fft * h

    # 对希尔伯特变换后的信号进行逆傅里叶变换
    y = torch.fft.ifft(y_fft)

    # 取实部作为输出
    y = y.real

    return y

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
x = x.unsqueeze(0)  # 在第0维上添加一个维度
h_x = hilbert_transform(x)
print(h_x)
