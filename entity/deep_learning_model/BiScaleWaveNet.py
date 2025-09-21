import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pywt


"""
结合小波变换的双流WaveNet结构

多尺度特征提取架构:
1.在输入预处理阶段引入离散小波变换（DWT），将原始信号分解为低频近似（LL）和高频细节（LH/HL/HH）分量
2.设计双流处理结构：低频分量走全局趋势捕捉通道，高频分量走局部细节分析通道
3.使用可学习小波基替代固定小波基（如Daubechies），通过反向传播优化基函数参数

可学习小波基：通过反向传播优化滤波器参数，比固定小波基更适应任务特性
异构正则化：高频路径使用Dropout，低频路径保留完整信息
膨胀卷积组合：兼顾长程依赖与局部细节，降低计算复杂度
"""


# 多尺度特征提取架构
class BiScaleWaveNet(nn.Module):

    """
        核心创新点：
        可学习小波分解：替代传统固定小波基（如Daubechies），通过反向传播优化滤波器参数
        双流异构处理：低频路径采用指数级膨胀卷积（1,2,4,8），高频路径采用几何级数膨胀率（1,3,9,27）
    """

    def __init__(self, input_dim, output_dim, num_scales=2):
        super().__init__()

        # 通道数
        in_channels = 1

        # 输入预处理
        self.dwt = LearnableDWT(in_channels=in_channels)  # 可学习离散小波变换DWT
        self.norm = nn.BatchNorm1d(4 * in_channels)

        # 双流处理路径
        self.low_stream = WaveletStream(4 * in_channels, 32, mode='low')  # 低频全局路径
        self.high_stream = WaveletStream(4 * in_channels, 32, mode='high')  # 高频细节路径

        # 动态特征融合
        self.attention = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.Sigmoid()
        )

        # 输出层
        self. final_conv = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # 输入形状: [B, Seq, Features]

        # 可学习小波分解
        x_dwt = self.dwt(x)  # [B, 4F, L//2]
        # x_dwt = x_dwt.transpose(1, 2)
        x_dwt = self.norm(x_dwt)

        # 双路径处理
        x_low = self.low_stream(x_dwt)  # 全局趋势路径
        x_high = self.high_stream(x_dwt)  # 局部细节路径

        # 动态特征融合
        fused = torch.cat([x_low, x_high], dim=1)
        attn = self.attention(fused)
        fused = fused * attn

        # 回归输出
        return self.final_conv(fused)


# 可学习小波变换层
class LearnableDWT(nn.Module):
    """可学习小波变换层（基于Daubechies基初始化）"""

    """
        关键技术:
        1.通过1D卷积实现DWT分解，输出包含LL/LH/HL/HH四个子带
        2.反射填充避免边界效应
    """

    # DWT: 离散小波变换

    def  __init__(self, in_channels, wave_type='db4'):
        super().__init__()

        # 初始化可训练小波滤波器
        wavelet = pywt.Wavelet(wave_type)  # 初始化Daubechies小波基
        dec_hi = torch.Tensor(wavelet.dec_hi)  # 高通滤波器
        dec_lo = torch.Tensor(wavelet.dec_lo)  # 低通滤波器

        # 注册为可学习参数
        self.register_buffer('initial_dec_hi', dec_hi)
        self.register_buffer('initial_dec_lo', dec_lo)

        self.dec_hi = Parameter(dec_hi.clone())
        self.dec_lo = Parameter(dec_lo.clone())

        # 通道卷积
        # 修改卷积层：输出通道=4*in_channels，分组=in_channels
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=4 * in_channels,
            kernel_size=len(wavelet.dec_lo),  # 设为8
            bias=False,
            stride=2,
            padding=3,  # 填充3保证输出长度=⌈1212/2⌉=606
            padding_mode='reflect',  # 反射填充保持边界特征[1](@ref)
            groups=1  # 分组卷积保持通道独立[7](@ref)
        )

    def forward(self, x):
        # 动态构造可学习滤波器组
        # 构造四通道滤波器组（LL/LH/HL/HH）
        filters = torch.cat([
            self.dec_lo.unsqueeze(0).repeat(1, 1, 1),  # LL
            self.dec_hi.unsqueeze(0).repeat(1, 1, 1),  # LH
            self.dec_lo.unsqueeze(0).repeat(1, 1, 1),  # HL
            self.dec_hi.unsqueeze(0).repeat(1, 1, 1)  # HH
        ], dim=0)  # 形状 [4, 1, 8]
        self.conv.weight = nn.Parameter(filters)  # 动态更新卷积核

        x = self.conv(x)
        return x


# 双流处理模块
class WaveletStream(nn.Module):
    """双流处理结构核心模块"""

    """
        设计亮点:
        1.异构膨胀策略:低频路径关注长程依赖(指数膨胀),高频路径捕捉局部突变(几何膨胀)
        2.高频路径正则化:引入Dropout防止过拟合
    """

    def __init__(self, in_dim, out_dim, mode='low'):
        super().__init__()
        self.mode = mode
        # 动态设置膨胀率
        dilation_rates = [1, 2, 4, 8] if mode == 'low' else [1, 3, 9, 27]

        # 多尺度膨胀卷积组
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_dim, out_dim, 3,
                          dilation=d, padding=d),
                nn.BatchNorm1d(out_dim),
                nn.GELU()  # 比ReLU更适合小波特征
            ) for d in dilation_rates
        ])

        # 跨尺度特征融合
        self.fuse_conv = nn.Conv1d(len(dilation_rates) * out_dim, out_dim, 1)

    def forward(self, x):
        features = []
        for conv in self.conv_blocks:
            x_conv = conv(x)
            if self.mode == 'high':
                x_conv = F.dropout(x_conv, 0.2)  # 高频路径增强正则化
            features.append(x_conv)
        return self.fuse_conv(torch.cat(features, dim=1))
