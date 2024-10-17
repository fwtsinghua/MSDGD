import torch
import torch.nn as nn

class ConditionNet(nn.Module):
    """
    A simple feedforward neural network for conditioning the diffusion process
    把自定义的表格条件转换为张量，嵌入到扩散过程中
    """
    def __init__(self, in_row, in_col, out_row, out_col):
        super(ConditionNet, self).__init__()
        self.fc1 = nn.Linear(in_row*in_col, 128)  # 第一层映射到一个较大的中间维度
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, out_row*out_col)  # 最后一层映射到目标维度
        self.out_row = out_row
        self.out_col = out_col

    def forward(self, x):
        nan_mask = torch.isnan(x)  # 检测 NaN 值
        x[nan_mask] = 0  # 填充 NaN 值为 0
        x_flatten = x.flatten()  # 展平张量
        x = torch.relu(self.fc1(x_flatten))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # 重新塑形输出以匹配目标张量的形状
        x = x.view(self.out_row, self.out_col)
        return x



x = torch.randn(16, 4)
net = CustomNet(16,4,96,40)


y = net(x)
print(y.shape)