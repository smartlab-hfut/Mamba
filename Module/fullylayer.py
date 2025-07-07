import torch
import torch.nn as nn

class FullyCon(nn.Module):
    """
    使用全连接层实现的简单分类器:
      输入: [B, L, d1, d2]
      输出: [B, num_classes]
    过程:
      1) flatten (L, d1, d2) => total_dim
      2) 全连接层 => 输出分类结果
    """
    def __init__(self, d1, d2, length, num_classes, hidden_dim=12):
        super(FullyCon, self).__init__()
        self.input_dim = length * d1 * d2  # 将 (L, d1, d2) 扁平化为一维
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),  # 隐藏层
            nn.ReLU(),                              # 激活函数
            nn.Linear(hidden_dim, num_classes)      # 输出层
        )

    def forward(self, x):
        """
        x: [B, L, d1, d2]
        返回: [B, num_classes]
        """
        B, L, D1, D2 = x.shape

        # flatten: [B, L, d1, d2] => [B, total_dim]
        x = x.view(B, -1)

        # 全连接分类
        logits = self.fc(x)  # [B, num_classes]
        return logits


