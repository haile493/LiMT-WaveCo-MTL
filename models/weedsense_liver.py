"""
WeedSense-3D adapted for Liver Multi-Task CT Dataset (LiMT).
Applies the core innovations of WeedSense (ICCV 2025) to 3D medical imaging:
1. 3D Dual-Path Encoder:
   - 3D Detail Branch: Shallow-wide architecture preserving liver boundaries and fine tumor margins.
   - 3D Semantic Branch: Deep-narrow architecture with 3D Universal Inverted Bottleneck (UIB-3D)
     blocks in S0-M3-E0 configuration with 3D Squeeze-and-Excitation (SE-3D) and Context Embedding.
2. 3D Bilateral Guided Aggregation (BGA-3D):
   - Fuses detail and semantic representations via cross-attention gating.
3. Multi-Task Bifurcated Decoder:
   - 3D Semantic Segmentation Head with deep multi-scale auxiliary supervision (Stem, S3, S4, S5).
   - Global Multi-Task Decoder with 8-Head Transformer Self-Attention (TGD-3D) predicting:
     * Multi-label lesion classification (5 classes: Liver, Cyst, Hemangioma, Metastasis, HCC).
     * Quantitative continuous tumor burden / volume percentage regression.
"""

import math
from typing import Tuple, Dict, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvModule3D(nn.Sequential):
    """Standard 3D Conv - InstanceNorm3d - LeakyReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act=True):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True)
        ]
        if act:
            layers.append(nn.LeakyReLU(0.01, inplace=True))
        super().__init__(*layers)


class SqueezeExcite3D(nn.Module):
    """3D Squeeze-and-Excitation block for adaptive volumetric channel recalibration."""
    def __init__(self, in_channels, rd_ratio=0.25):
        super().__init__()
        rd_channels = make_divisible(int(in_channels * rd_ratio), 4)
        self.conv_reduce = nn.Conv3d(in_channels, rd_channels, 1, bias=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.conv_expand = nn.Conv3d(rd_channels, in_channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        # Global 3D pooling across (D, H, W)
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class LayerScale3D(nn.Module):
    """LayerScale 3D for numerical stabilization in deep inverted bottlenecks."""
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma.view(1, -1, 1, 1, 1)


class UIBBlock3D(nn.Module):
    """
    3D Universal Inverted Bottleneck (UIB-3D) Block.
    Optimal configuration S0-M3-E0:
    - 1x1x1 pointwise expansion
    - 3x3x3 middle depthwise convolution
    - 3D Squeeze-and-Excitation (SE-3D)
    - 1x1x1 pointwise projection
    - LayerScale3D + residual connection
    """
    def __init__(self, in_channels, out_channels, stride=1, exp_ratio=4, se_ratio=0.25):
        super().__init__()
        self.has_skip = (stride == 1 and in_channels == out_channels)
        mid_channels = make_divisible(int(in_channels * exp_ratio), 4)

        # 1. Pointwise expansion
        self.pw_exp = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 1, bias=False),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # 2. Middle depthwise 3x3x3 convolution
        self.dw_mid = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # 3. Squeeze-and-Excitation
        self.se = SqueezeExcite3D(mid_channels, rd_ratio=se_ratio)

        # 4. Pointwise projection
        self.pw_proj = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, 1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True)
        )

        # 5. Layer scale
        self.layer_scale = LayerScale3D(out_channels)

    def forward(self, x):
        shortcut = x
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = x + shortcut
        return x


class StemBlock3D(nn.Module):
    """
    3D Stem block for initial feature extraction.
    Downsamples by 4x while preserving rich contrast gradients.
    """
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.conv1 = ConvModule3D(in_channels, out_channels, 3, stride=2, padding=1)  # D/2
        self.left = nn.Sequential(
            ConvModule3D(out_channels, out_channels // 2, 1, stride=1, padding=0),
            ConvModule3D(out_channels // 2, out_channels, 3, stride=2, padding=1)
        )
        self.right = nn.MaxPool3d(3, stride=2, padding=1)
        self.fuse = ConvModule3D(out_channels * 2, out_channels, 3, stride=1, padding=1)  # D/4

    def forward(self, x):
        x = self.conv1(x)
        x_left = self.left(x)
        x_right = self.right(x)
        return self.fuse(torch.cat([x_left, x_right], dim=1))


class ContextEmbeddingBlock3D(nn.Module):
    """3D Context Embedding block for capturing global volume context."""
    def __init__(self, channels=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.conv2 = ConvModule3D(channels, channels, 3, stride=1, padding=1)

    def forward(self, x):
        y = self.conv1(self.pool(x))
        return self.conv2(x + y)


class DetailBranch3D(nn.Module):
    """
    3D Detail Branch (Shallow-Wide):
    Preserves liver capsule edges, organ borders, and subtle lesion margins.
    S1 (D/2, 32) -> S2 (D/4, 32) -> S3 (D/8, 64).
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.S1 = nn.Sequential(
            ConvModule3D(in_channels, 32, 3, stride=2, padding=1),
            ConvModule3D(32, 32, 3, stride=1, padding=1)
        )
        self.S2 = nn.Sequential(
            ConvModule3D(32, 32, 3, stride=2, padding=1),
            ConvModule3D(32, 32, 3, stride=1, padding=1),
            ConvModule3D(32, 32, 3, stride=1, padding=1)
        )
        self.S3 = nn.Sequential(
            ConvModule3D(32, 64, 3, stride=2, padding=1),
            ConvModule3D(64, 64, 3, stride=1, padding=1),
            ConvModule3D(64, 64, 3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.S3(self.S2(self.S1(x)))  # (B, 64, D/8, H/8, W/8)


class SemanticBranch3D(nn.Module):
    """
    3D Semantic Branch (Deep-Narrow):
    Aggressive hierarchical downsampling with UIB-3D blocks for high-level semantic lesion context.
    - Stem: D/4, 16 ch
    - S3: D/8, 32 ch
    - S4: D/16, 64 ch
    - S5: D/32, 64 ch + Context Embedding Block
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = StemBlock3D(in_channels, out_channels=16)
        self.S3 = nn.Sequential(
            UIBBlock3D(16, 32, stride=2, exp_ratio=4),
            UIBBlock3D(32, 32, stride=1, exp_ratio=4)
        )
        self.S4 = nn.Sequential(
            UIBBlock3D(32, 64, stride=2, exp_ratio=4),
            UIBBlock3D(64, 64, stride=1, exp_ratio=4)
        )
        self.S5_1 = nn.Sequential(
            UIBBlock3D(64, 64, stride=2, exp_ratio=4),
            UIBBlock3D(64, 64, stride=1, exp_ratio=4),
            UIBBlock3D(64, 64, stride=1, exp_ratio=4)
        )
        self.S5_2 = ContextEmbeddingBlock3D(64)

    def forward(self, x):
        x2 = self.stem(x)       # D/4, 16
        x3 = self.S3(x2)        # D/8, 32
        x4 = self.S4(x3)        # D/16, 64
        x5_1 = self.S5_1(x4)    # D/32, 64
        x5_2 = self.S5_2(x5_1)  # D/32, 64
        return x2, x3, x4, x5_1, x5_2


class BGALayer3D(nn.Module):
    """
    3D Bilateral Guided Aggregation Layer:
    Fuses spatial detail features (D/8) with semantic features (D/32) using cross-attention gating.
    """
    def __init__(self, channels=64):
        super().__init__()
        self.left1 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.Conv3d(channels, channels, 1, 1, 0, bias=False)
        )
        self.left2 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 2, 1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.AvgPool3d(3, 2, 1, count_include_pad=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
            nn.Sigmoid()
        )
        self.right2 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.Conv3d(channels, channels, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.up = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        self.fuse_conv = ConvModule3D(channels, channels, 3, 1, 1)

    def forward(self, x_d, x_s):
        x1 = self.left1(x_d)
        x2 = self.left2(x_d)
        x3 = self.right1(x_s)
        x4 = self.right2(x_s)

        left = x1 * x3
        right = self.up(x2 * x4)
        return self.fuse_conv(left + right)


class SegmentationHead3D(nn.Module):
    """3D Segmentation Head with progressive upsampling to full resolution."""
    def __init__(self, in_channels=64, mid_channels=128, num_classes=6, upscale_factor=8, is_aux=False):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = ConvModule3D(in_channels, mid_channels, 3, 1, 1)
        self.dropout = nn.Dropout3d(0.1)

        if upscale_factor == 8:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(mid_channels, 64, kernel_size=2, stride=2, bias=False),
                nn.InstanceNorm3d(64, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
                nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, bias=False),
                nn.InstanceNorm3d(32, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
                nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, bias=False),
                nn.InstanceNorm3d(16, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(16, num_classes, kernel_size=1)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=upscale_factor, mode='trilinear', align_corners=False),
                nn.Conv3d(mid_channels, num_classes, kernel_size=1)
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        return self.up(x)


class TransformerBlock1D(nn.Module):
    """Transformer block for the Global Multi-Task Decoder (TGD-3D)."""
    def __init__(self, embed_dim=512, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TemporalGrowthDecoder3D(nn.Module):
    """
    3D Temporal Growth Decoder (TGD-3D) adapted for Liver Multi-Task Learning.
    Processes aggregated volumetric features through:
    1. 3D Global Average Pooling -> 64-dim representation
    2. Linear projection to 512-dim embedding
    3. Multi-Head Self-Attention Transformer block (8 heads, 2048 FFN, GELU)
    4. Shared task-specific MLP
    5. Dual task output heads:
       - Multi-label Lesion Classification (5 classes)
       - Quantitative Tumor Burden Regression (continuous volume ratio %)
    """
    def __init__(self, in_channels=64, embed_dim=512, num_heads=8, num_classes=5):
        super().__init__()
        self.project = nn.Linear(in_channels, embed_dim)
        self.transformer = TransformerBlock1D(embed_dim=embed_dim, num_heads=num_heads, dim_feedforward=2048, dropout=0.1)

        self.task_head = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
        )

        self.cls_output = nn.Linear(512, num_classes)  # Multi-label classification (5 classes)
        self.burden_output = nn.Linear(512, 1)          # Continuous tumor burden % regression
        self.dropout = nn.Dropout(0.3)

    def forward(self, fused_features):
        # Global 3D average pooling
        global_feat = F.adaptive_avg_pool3d(fused_features, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        global_feat = self.dropout(global_feat)

        proj = self.project(global_feat).unsqueeze(1)  # (B, 1, embed_dim)
        trans = self.transformer(proj).squeeze(1)      # (B, embed_dim)
        task_feat = self.task_head(trans)

        cls_logits = self.cls_output(task_feat)
        tumor_burden_pred = self.burden_output(task_feat)
        return cls_logits, tumor_burden_pred


class WeedSenseLiverNet(nn.Module):
    """
    WeedSense Architecture for 3D Liver Multi-Task Analysis.
    Jointly performs:
    1. 3D Liver and Lesion Multi-Class Segmentation (Background, Liver, Cyst, Hemangioma, Metastasis, HCC)
    2. Multi-Label Lesion Classification (Liver, Cyst, Hemangioma, Metastasis, HCC)
    3. Quantitative Tumor Burden Percentage Regression
    """
    def __init__(self, in_channels=1, num_seg_classes=6, num_cls_classes=5):
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes

        # Dual-Path 3D UIB Encoder
        self.detail_branch = DetailBranch3D(in_channels=in_channels)
        self.semantic_branch = SemanticBranch3D(in_channels=in_channels)
        self.aggregation_layer = BGALayer3D(channels=64)

        # Primary Segmentation Head
        self.seg_head = SegmentationHead3D(in_channels=64, mid_channels=128, num_classes=num_seg_classes, upscale_factor=8, is_aux=False)

        # Auxiliary Segmentation Heads for training-only deep supervision
        self.aux_stem = SegmentationHead3D(in_channels=16, mid_channels=32, num_classes=num_seg_classes, upscale_factor=4, is_aux=True)
        self.aux_s3 = SegmentationHead3D(in_channels=32, mid_channels=32, num_classes=num_seg_classes, upscale_factor=8, is_aux=True)
        self.aux_s4 = SegmentationHead3D(in_channels=64, mid_channels=64, num_classes=num_seg_classes, upscale_factor=16, is_aux=True)
        self.aux_s5 = SegmentationHead3D(in_channels=64, mid_channels=64, num_classes=num_seg_classes, upscale_factor=32, is_aux=True)

        # Global Multi-Task Decoder with Self-Attention Transformer
        self.tgd = TemporalGrowthDecoder3D(in_channels=64, embed_dim=512, num_heads=8, num_classes=num_cls_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.InstanceNorm3d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 1. Dual-Path Encoder
        x_d = self.detail_branch(x)                        # (B, 64, D/8, H/8, W/8)
        aux2, aux3, aux4, aux5, x_s = self.semantic_branch(x)  # x_s: (B, 64, D/32, H/32, W/32)

        # 2. Bilateral Guided Aggregation
        fused = self.aggregation_layer(x_d, x_s)           # (B, 64, D/8, H/8, W/8)

        # 3. Task Decoders
        seg_logits = self.seg_head(fused)                  # (B, 6, D, H, W)
        cls_logits, tumor_burden = self.tgd(fused)         # (B, 5), (B, 1)

        out = {
            'seg_logits': seg_logits,
            'cls_logits': cls_logits,
            'tumor_burden': tumor_burden
        }

        # During training, provide deep auxiliary supervision
        if self.training:
            out['aux_stem'] = self.aux_stem(aux2)
            out['aux_s3'] = self.aux_s3(aux3)
            out['aux_s4'] = self.aux_s4(aux4)
            out['aux_s5'] = self.aux_s5(aux5)

        return out
