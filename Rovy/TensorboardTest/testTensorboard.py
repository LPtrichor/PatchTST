import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 创建Tensorboard的SummaryWriter对象
writer = SummaryWriter('logs')

# 定义训练数据
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)

# 定义模型、损失函数和优化器
model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练过程
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录训练过程中的指标
    writer.add_scalar('Loss/train', loss.item(), epoch)
# 指定曲线颜色
# color_scheme = {
#         'Loss/train': 'red'
#     }
# writer.add_custom_scalars(color_scheme)
# 关闭SummaryWriter对象
writer.close()
