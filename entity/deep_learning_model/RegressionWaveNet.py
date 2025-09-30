import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class RegressionWaveNet(nn.Module):
    def __init__(self, input_dim, output_dim):

        super().__init__()

        # 模型参数
        self.layers = 6
        self.blocks = 3
        self.dilation_channels = 16
        self.residual_channels = 16
        self.skip_channels = 64
        self.kernel_size = 2

        # 输入标准化
        self.input_norm = nn.BatchNorm1d(input_dim)

        # 初始卷积
        self.start_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.residual_channels,
            kernel_size=1
        )

        # 构建膨胀卷积层堆
        self.dilations = self._generate_dilations()
        self.res_blocks = nn.ModuleList([
            self._build_res_block(i)
            for i in range(len(self.dilations))
        ])

        # 输出层
        self.final_conv = nn.Conv1d(
            in_channels=self.residual_channels,
            out_channels=self.skip_channels,
            kernel_size=1
        )
        self.final_dense = nn.Linear(self.skip_channels, output_dim)

        # 正则化
        self.dropout = nn.Dropout(0.2)

    def _generate_dilations(self):
        """生成指数增长的膨胀率序列"""
        return [2 ** (i % 6) for _ in range(self.blocks)
                for i in range(self.layers)]

    def _build_res_block(self, layer_idx):
        """构建单个残差块"""
        # 替换膨胀卷积为普通卷积（kernel_size=1）
        return nn.Sequential(
            # nn.Conv1d(
            #     in_channels=self.residual_channels,
            #     out_channels=self.dilation_channels,
            #     kernel_size=self.kernel_size,
            #     dilation=self.dilations[layer_idx],
            #     padding=(self.kernel_size - 1) * self.dilations[layer_idx]
            # ),
            nn.Conv1d(
                in_channels=self.residual_channels,
                out_channels=self.dilation_channels,
                kernel_size=1,  # 无时间维度操作
                dilation=1
            ),
            # nn.LayerNorm(self.dilation_channels),
            nn.BatchNorm1d(self.dilation_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(
                in_channels=self.dilation_channels,
                out_channels=self.residual_channels,
                kernel_size=1
            )
        )

    def forward(self, x):
        # 输入标准化 [B, Features, Seq]
        x = x.permute(0, 2, 1)
        x = self.input_norm(x)

        # 初始卷积
        x = self.start_conv(x)

        # 残差连接
        skip_connections = []
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual  # 残差连接
            skip_connections.append(x[:, :, -1:])  # 取最后时间步

        # 聚合特征
        x = torch.cat(skip_connections, dim=-1)
        x = self.final_conv(x)

        # 全局平均池化 + 回归输出
        x = x.mean(dim=-1)
        x = self.final_dense(x)
        return x
