import torch.nn as nn
import torch
import torch.nn.functional as F  # 包含激活函数、损失函数等操作
from einops import rearrange  # einops用于张量操作的便捷工具
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out  # 计算神经网络层的输入输出通道数，有助于初始化
from pdb import set_trace as stx


# import cv2

# 一、参数初始化函数
# （1）截断正态分布初始化
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


# 二、辅助模块（预归一化层）： 在输入数据送入某个模块fn之前先进行LayerNorm标准化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# GELU激活函数，比 ReLU 更平滑
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


# 卷积层封装，封装标准卷积，自动计算padding，避免输入输出尺寸变化
def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


# 三、光照估计器模块Illumination_Estimator（计算亮度信息，生成illu_map用于增强）
# 1.特征提取：通过 1x1 卷积将输入图像的通道数映射到中间特征空间
# 2.局部特征增强：通过分组深度卷积提取局部光照特征。
# 3.光照图生成：通过 1x1 卷积生成 RGB 光照图，指导后续的图像亮度调整。
class Illumination_Estimator(nn.Module):
    def __init__(
            # n_fea_middle中间特征通道数，通常用来提取特征 3
            # n_fea_in输入通道数，由于 mean_c 计算得到的单通道图像Lp会与原图I拼接，形成 4 通道输入。
            # n_fea_out输出通道数，对应 RGB 图像的 3 个通道
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()
        # 第一层卷积：通道变换：特征映射，nn.Conv2d二维卷积层用于特征提取。使用 1x1 卷积，不改变空间维度，仅通过线性变换调整通道数。n_fea_in=4输入通道数
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        # 深度卷积：局部特征提取，增强光照特征Flu，使用 5x5 的卷积核，捕捉更大的局部信息。padding保持输入尺寸与输出相同，边缘填充2个像素
        # groups=n_fea_in：进行分组卷积。每组卷积独立操作在一部分通道上，有效减少计算量，同时增强特征独立性。将特征图分为n_fea_in组，每组进行独立卷积
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        # 第二层卷积：生成光照图Lbar，通过线性变换，将特征映射为光照图，光照图的值通常在 [0, 1] 之间
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        # 将原图I的 RGB 通道与亮度Lp均值通道融合，增强光照信息
        mean_c = img.mean(dim=1).unsqueeze(1)  # 图像在通道维度（RGB通道）上取平均值，得到灰度图Lp
        # stx()
        input = torch.cat([img, mean_c], dim=1)  # 将原图I和灰度图Lp在通道维度上拼接（由3通道变为4通道）

        # 第一层卷积执行：输入是经过拼接后的4通道特征图，输出中间特征图，通道数变为n_fea_middle=3
        x_1 = self.conv1(input)
        # 使用 5x5 深度卷积核提取局部特征，提取光照特征Flu。输入中间特征图，输出Flu
        illu_fea = self.depth_conv(x_1)
        # 生成 3 通道的光照图Lbar，用于调整图像亮度。输入Flu，输出Lbar
        illu_map = self.conv2(illu_fea)
        # 最终输出：Flu（光照特征，座位输入到IGAB模块，引导注意力机制）和Lbar（光照图，用于图像增强和亮度调整）
        return illu_fea, illu_map  # Flu提供局部和全局光照特征信息 Lbar通过像素级调整提升图像亮度


# 四、光照引导多头自注意力模块（IG_MSA)
class IG_MSA(nn.Module):  # dim输入特征的通道维度。 dim_head每个注意力头的特征维度，默认为 64 。heads注意力头的数量，默认为 8
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):  # 构建一个多头自注意力模块，通过光照特征illu_fea_trans，引导注意力计算
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        # 参数初始化 1.生成Q、K、V矩阵的线性层
        # nn.Linear：对输入特征进行线性变换，将原始特征映射为 Query（Q）、Key（K）和 Value（V）矩阵。
        # 输入维度dim转换为多头特征维度，每个头的维度为dim_head，总特征维度为dim_head * heads
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)  # 不使用偏置项bias，因为不需要额外的偏移量
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        # 缩放参数nn.Parameter，创建一个可训练的参数，形状为(heads, 1, 1)，torch.ones：初始值为 1，用于在训练过程中学习不同头的缩放系数
        # 在注意力矩阵计算时，对每个注意力头的结果进行缩放。   （应该是W：表示fc层的可学习参数）
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        # 输出投影层 将多头注意力的输出dim_head * heads映射回原始特征维度dim。包含偏置项，用于进一步优化输出
        # 恢复原始特征维度，方便后续网络处理
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        # 位置编码层 nn.Sequential将一系列操作封装为一个顺序执行的模块
        # 提供位置编码信息，弥补注意力机制中缺少位置信息的问题 ？？？？？、
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),  # 使用 3x3 卷积提取局部空间信息
            GELU(),  # 激活函数，具有更平滑的曲线，有助于特征学习
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):  # illu_fea_trans：经过特征变换？？？？？、、、？的光照特征图(变换的Flu)
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c] #代表批量大小、高度、宽度、通道数
        """
        b, h, w, c = x_in.shape  # x_in:输入特征图Fin，形状为(batch_size, height, width, channels)
        # 矩阵展平操作：为后续线性层处理提供合适的输入形状
        x = x_in.reshape(b, h * w, c)  # 将输入特征图Fin展平成二维矩阵，h * w表示每个图像的像素总数，高度乘以宽度
        # 生成Q、K、V矩阵 通过线性层fc？？？从输入特征中生成，生成后形状为(b, hw, heads * dim_head)
        # 功能：将输入特征映射为注意力计算所需的 Q、K、V 矩阵
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # 光照引导的特征处理：将光照特征映射Flu到与特征图Fin相同的形状，用于后续注意力引导
        # 将光照特征Flu映射到与特征图Fin相同的形状，用于后续注意力引导
        illu_attn = illu_fea_trans  # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))  # 光照特征Fin也被划分为多头形式
        # rearrange：将输入特征划分为多头形式 从(b, hw, heads * dim_head)->(b, heads, hw, dim_head)
        v = v * illu_attn  # 让光照特征Fin直接影响注意力计算：使用光照特征引导 Value 矩阵，强化光照信息
        # q: b,heads,hw,c
        # 维度转置：准备注意力计算：为注意力分数计算做准备，使得矩阵乘法维度匹配
        q = q.transpose(-2, -1)  # transpose(-2, -1)交换最后两个维度
        k = k.transpose(-2, -1)  # 矩阵乘法中，需要K的最后一个维度和Q的倒数第二个维度进行乘法
        v = v.transpose(-2, -1)
        # 归一化：防止数值过大。F.normalize对矩阵进行L2归一化 （引入a）
        q = F.normalize(q, dim=-1, p=2)  # dim=-1：在最后一个维度上进行归一化，即对每个特征向量
        k = F.normalize(k, dim=-1, p=2)  # p=2：L2范数，然后再归一化，防止数值过大导致梯度爆炸，同时保留方向信息，增强训练的稳定性
        # 注意力计算：生成注意力权重，用于后续特征加权
        # 计算K和Q之间的相似度
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q 。A：注意力分数矩阵，表示每个像素之间的相关性
        # 缩放注意力分数 self.rescale(heads, 1, 1)是一个可训练参数，用于缩放注意力分数
        # 功能：提供额外的可学习参数，调整不同头的注意力分数范围，进一步增强模型性能a
        attn = attn * self.rescale
        # 归一化注意力分数：在最后一个维度（hw维度）上进行 Softmax 操作：将注意力分数归一化为概率分布，方便后续加权
        attn = attn.softmax(dim=-1)
        # 应用注意力权重：使用注意力权重对 v = v * illu_attn进行加权求和
        x = attn @ v  # b,heads,d,hw
        # 转置回原始形状，permute：调整维度顺序，为后续线性映射做准备
        x = x.permute(0, 3, 1, 2)  # Transpose
        # 重塑输出，将多头输出拼接回原始特征维度c=self.num_heads * self.dim_head
        # 将多头注意力结果合并回单一特征图
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        # 线性映射回原始维度：通过线性层fc将特征图映射回原始维度（恢复特征图的空间分辨率）
        out_c = self.proj(x).view(b, h, w, c)
        # 位置编码p补充信息，增强特征表达能力
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        # 特征加法，将注意力输出和位置编码结果融合，生成最终的光照引导注意力特征图
        out = out_c + out_p

        return out

        # 五、前馈网络模块FFN：用于在 Transformer 框架内对特征进行非线性变换和增强1*1-3*3-1*1的卷积结构
        # 通过非线性变换捕捉更加复杂的特征和表示
        # class FeedForward(nn.Module):
        #    def __init__(self, dim, mult=4):
        #        super().__init__()
        #       self.net = nn.Sequential( #Sequential由三层卷积 + GELU激活函数组成
        # 第一层：通道扩展
        #            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
        #            GELU(),#非线性变换。是一种比 ReLU 更平滑的非线性激活函数，适用于 Transformer 模型
        # 第二层：深度卷积3*3的增加局部感受野（局部特征提取）
        #            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
        #                      bias=False, groups=dim * mult),
        #            GELU(),
        # 第三层：通道恢复
        #            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        #        )

        #    def forward(self, x):#前向传播部分
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """


#       out = self.net(x.permute(0, 3, 1, 2).contiguous())#调整张量格式，前馈计算
#       return out.permute(0, 2, 3, 1)#恢复维度顺序


# --------------------我的改进FFN
# 轻量通道注意力：ECA
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)


# 轻量卷积模块
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# 改进后的前馈模块
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.SiLU(),

            DepthwiseSeparableConv2d(dim * mult, dim * mult, 3, 1, 1),
            nn.SiLU(),

            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

        self.eca = ECABlock(dim)

    def forward(self, x):  # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous()  # -> [B, C, H, W]
        out = self.net(x)
        out = self.eca(out)
        return out.permute(0, 2, 3, 1).contiguous()  # -> [B, H, W, C]


# 光照引导注意力块（IGAB)
# 主要用于特征聚合，结合了Transformer机制的自注意力（IGMSA）和 前馈网络（FeedForward），以增强全局信息建模能力。
class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,  # 默认2个block子模块（attention+FFN）
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),  # 1.多头自注意力模块（IGMSA），负责计算全局自注意力，提取长距离依赖。
                PreNorm(dim, FeedForward(dim=dim))  # 2.归一化后的前馈网络，增强特征表达。
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)  # [b, h, w, c]
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x  # 自注意力+残差连接
            x = ff(x) + x  # 前馈网络+残差连接
        out = x.permute(0, 3, 1, 2)  # [b, c, h, w]
        return out


# 七、去噪器模块（上采样和下采样） IGT恢复器
class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection 输入投影（Embedding 层）
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder 编码器
        self.encoder_layers = nn.ModuleList([])  # 由2个IGAB+降采样组合构成
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(  # 采用IGAB计算注意力，提取全局特征 4×4 步长为2 的卷积实现降采样（下采样/池化），通道数翻倍
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),  # 降采样
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)  # 光照信息降采样Flu
            ]))
            dim_level *= 2

        # Bottleneck瓶颈层
        self.bottleneck = IGAB(  # 在最深层进行自注意力建模（全局特征提取），通过编码器放大通道数
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder解码器
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):  # ConvTranspose2d反卷积实现上采样，恢复空间分辨率
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),  # 1*1卷积融合（跳跃连接时，拼接的特征维度会翻倍）
                IGAB(  # 在上采样后进行 自注意力计算，提高重建质量
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection 输出映射 。3x3 卷积将特征通道转换回 RGB 3 通道
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # 激活函数，防止梯度消失
        self.apply(self._init_weights)

    def _init_weights(self, m):  # 初始化权重，防止梯度爆炸
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):  # 前向传播
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding 输入投影
        fea = self.embedding(x)

        # Encoder 编码器（下采样）降低分辨率，提高通道数
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea, illu_fea)  # bchw 计算自注意力
            illu_fea_list.append(illu_fea)  # 记录光照特征
            fea_encoder.append(fea)  # 记录特征
            fea = FeaDownSample(fea)  # 降采样
            illu_fea = IlluFeaDownsample(illu_fea)  # 降采样光照特征

        # Bottleneck瓶颈层
        fea = self.bottleneck(fea, illu_fea)  # 特征增强，经过IGAB进行最深层全局建模

        # Decoder解码器（上采样）
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)  # 上采样
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))  # 跳跃连接融合
            illu_fea = illu_fea_list[self.level - 1 - i]  # 取出对应的光照特征
            fea = LeWinBlcok(fea, illu_fea)  # 计算自注意力

        # Mapping输出映射
        out = self.mapping(fea) + x  # 残差连接，保留原始信息，增加去噪效果

        return out


# 单阶段的retinex
class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        # 估计光照分布 负责从输入图像中提取光照信息 输出Flu和Lbar
        self.estimator = Illumination_Estimator(n_feat)
        # 基于光照信息去噪并增强图像（基于Flu进行去噪）光照引导？
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                 num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img

    def forward(self, img):
        # img:        b,c=3,h,w  低光图像

        # illu_fea:   b,c,h,w    Flu
        # illu_map:   b,c=3,h,w  Lbar

        illu_fea, illu_map = self.estimator(img)  # 输入I和Lp 输出Flu和Lbar
        # 解释一下公式不同：直接使用img * illu_map可能会因为光照映射的取值范围，导致某些区域的像素值变得非常小甚至为零，从而丢失图像的细节信息
        # 而加上原始图像 img 可以保留部分原始图像的信息，避免过度调整导致的信息丢失
        input_img = img * illu_map + img  # 增强后的图像Ilu=原图I*Lbar ？？、、？、、？？
        # 输入Flu和Ilu输出增强后Ien
        output_img = self.denoiser(input_img, illu_fea)

        return output_img


class RetinexFormer(nn.Module):  # 由多个RetinexFormer_Single_Stage组成
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 1, 1]):
        super(RetinexFormer, self).__init__()
        self.stage = stage
        # 3 个增强阶段，每个阶段都包含光照估计和图像去噪
        modules_body = [
            RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2,
                                       num_blocks=num_blocks)
            for _ in range(stage)]
        # 将所有RetinexFormer_Single_Stage串联起来，使得模型能逐步增强图像。
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):  # 逐个 RetinexFormer_Single_Stage 处理图像
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        return out

# if __name__ == '__main__':
#     from fvcore.nn import FlopCountAnalysis
#     model = RetinexFormer(stage=1,n_feat=40,num_blocks=[1,2,2]).cuda()
#     print(model)
#     inputs = torch.randn((1, 3, 256, 256)).cuda()
#     flops = FlopCountAnalysis(model,inputs)
#     n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
#     print(f'GMac:{flops.total()/(1024*1024*1024)}')
#     print(f'Params:{n_param}')