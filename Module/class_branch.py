import torch.fft
import torch
import torch.nn as nn
import torch.fft

class Classifier(nn.Module):
    def __init__(self, d1, d2, length, num_classes, hidden_dim=328):
        super(Classifier, self).__init__()

        # 特征降维：从 d1 和 d2 提取重要特征
        self.weight_d2 = nn.Parameter(torch.randn(d2, 1))  # d2 -> 1
        self.weight_d1 = nn.Parameter(torch.randn(d1, 1))  # d1 -> 1

        # 特征映射
        self.time_mapping = nn.Linear(length, hidden_dim)
        self.freq_mapping = nn.Linear(length, hidden_dim)

        # 融合层
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)

        # 分类层
        self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim//2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim//2, hidden_dim//4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim //4, num_classes)
                )

    def forward(self, x):
        B, L, d1, d2 = x.shape

        # Step 1: 对 d2 降维
        d2_weights = torch.softmax(self.weight_d2, dim=0)  # d2 -> 1
        reduced_d2 = torch.matmul(x, d2_weights).squeeze(-1)  # [B, L, d1]

        # Step 2: 对 d1 降维
        d1_weights = torch.softmax(self.weight_d1, dim=0)  # d1 -> 1
        reduced_features = torch.matmul(reduced_d2, d1_weights).squeeze(-1)  # [B, L]

        # Step 3: 提取频域特征
        fft_features = torch.fft.fft(reduced_features, dim=1)
        fft_magnitude = torch.abs(fft_features)  # 幅值 [B, L]
        fft_phase = torch.angle(fft_features)    # 相位 [B, L]

        # Step 4: 特征映射
        # time_features = self.time_mapping(reduced_features)  # 时域特征 [B, hidden_dim]
        freq_features = self.freq_mapping(fft_magnitude)     # 频域特征 [B, hidden_dim]\

        combined_features = freq_features
        # Step 5: 融合特征
        # combined_features = torch.cat([reduced_features, fft_magnitude], dim=-1)  # [B, hidden_dim * 2]
        # fused_features = self.fusion_layer(combined_features)  # [B, hidden_dim]

        # Step 6: 分类
        logits = self.classifier(combined_features)  # [B, num_classes]

        return logits




