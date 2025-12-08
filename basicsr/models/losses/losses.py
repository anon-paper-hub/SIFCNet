import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import kornia

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class LabColorLoss(nn.Module):
    """Color loss in Lab space (only a/b channels to reduce color bias).

    Args:
        loss_weight (float): Weight for color loss. Default: 1.0.
        reduction (str): Reduction mode. Supported: 'none' | 'mean' | 'sum'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(LabColorLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): (N, C, H, W), predicted RGB image in [0,1].
            target (Tensor): (N, C, H, W), ground truth RGB image in [0,1].
        """
        # 1. 先裁剪到安全区间<br/>
        pred = pred.clamp(1e-4, 1 - 1e-4)
        target = target.clamp(1e-4, 1 - 1e-4)

        # Convert to Lab color space
        pred_lab = kornia.color.rgb_to_lab(pred)
        target_lab = kornia.color.rgb_to_lab(target)

        # Only use a/b channels (ignore luminance L channel)
        pred_ab = pred_lab[:, 1:, :, :]
        target_ab = target_lab[:, 1:, :, :]

        loss = F.l1_loss(pred_ab, target_ab, reduction=self.reduction)
        return self.loss_weight * loss

#结构相似损失（对细节纹理边缘更敏感）
class SSIMLoss(nn.Module):
    """SSIM loss for image structural similarity.

    Args:
        loss_weight (float): Weight for SSIM loss. Default: 1.0.
        reduction (str): Reduction mode. Supported: 'none' | 'mean' | 'sum'.
        window_size (int): Size of the gaussian window. Default: 11.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', window_size=11):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.window_size = window_size

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): (N, C, H, W), predicted RGB image in [0,1].
            target (Tensor): (N, C, H, W), ground truth RGB image in [0,1].
        """
        # 1. 裁剪到安全区间，避免数值不稳定
        pred = pred.clamp(1e-4, 1 - 1e-4)
        target = target.clamp(1e-4, 1 - 1e-4)

        # 2. 计算 SSIM（kornia 实现，返回 [-1,1]）
        ssim_val = kornia.losses.ssim_loss(pred, target, window_size=self.window_size, reduction=self.reduction)

        # 注意：ssim_loss = 1 - SSIM，所以这里直接作为损失
        loss = self.loss_weight * ssim_val
        return loss
#语义感知损失
class _SemanticFeatureExtractor(nn.Module):
    """等价于你提供的 SemanticExtractor（两层 3x3 Conv + ReLU），
    作为感知特征提取器使用。"""
    def __init__(self, in_channels=3, n_feat=31):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, n_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)

class SemanticPerceptualLoss(nn.Module):
    """Semantic-aware perceptual loss using a lightweight feature extractor.

    公式： L_sem = || φ(pred) - φ(gt) ||_1 / (N * C * H * W)
    其中 φ 为固定（冻结参数）的语义特征提取网络。

    Args:
        loss_weight (float): 权重，默认 1.0（建议从 0.02~0.1 区间试）
        reduction (str): 'none' | 'mean' | 'sum'
        in_channels (int): 输入通道，默认 3
        n_feat (int): 语义特征维度，默认 31
    """
    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 in_channels=3,
                 n_feat=31):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f"Supported ones are: ['none','mean','sum']")
        self.loss_weight = loss_weight
        self.reduction = reduction

        # 构建与 SemanticExtractor 等价的编码器，作为固定特征提取器
        self.feat_net = _SemanticFeatureExtractor(in_channels=in_channels, n_feat=n_feat)
        # 冻结参数（但不阻断对输入的梯度）
        for p in self.feat_net.parameters():
            p.requires_grad = False
        self.feat_net.eval()

    @torch.no_grad()
    def _safe_clamp(self, x):
        # 仅用于数值安全裁剪，不参与图的构建
        return x.clamp_(1e-4, 1 - 1e-4)

    def forward(self, pred, target, **kwargs):
        """
        pred/target: (N, C, H, W), RGB in [0,1]
        """
        # 1) 数值安全：克隆后做有梯度的 clamp，保证梯度对 pred 仍可回传
        pred   = pred.clone()
        target = target.clone()
        pred   = torch.clamp(pred,   1e-4, 1 - 1e-4)
        target = torch.clamp(target, 1e-4, 1 - 1e-4)

        # 2) 通过冻结的特征网络提取特征（不使用 no_grad，让梯度能从 pred 传回）
        #    注：虽然 feat_net 的参数不更新，但算子仍可对输入求导，从而优化生成器
        pred_feat   = self.feat_net(pred)
        target_feat = self.feat_net(target)

        # 3) 计算 L1 特征差异
        loss = F.l1_loss(pred_feat, target_feat, reduction=self.reduction)
        return self.loss_weight * loss



