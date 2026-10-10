"""
WaveAttn-SOPSeg & Co-Seg Combined Architecture for 3D Volumetric Liver & Tumor CT Multi-Task Learning.

Adapted from:
1. WaveAttn-SOPSeg (from `cangio` project: `cangio/src/sopseg_enhanced.py`, integrating SAM2-CD WaveFusion + GLCAM + SOPSeg PRM):
   - 3D Discrete Wavelet Transform (HaarDWT3D / HaarIDWT3D):
     Decomposes 3D feature representations into 1 Low-Frequency semantic subband (LLL)
     and 7 High-Frequency directional edge/boundary subbands (LLH, LHL, LHH, HLL, HLH, HHL, HHH).
   - WaveFusion Refinement Block 3D:
     Separately processes low-frequency organ semantics and high-frequency lesion boundaries via dedicated residual streams,
     preventing boundary blurring of small focal liver lesions (Cyst, Hemangioma, Metastasis, HCC).
   - GLCAM-3D (Global-Local Context Attention Module 3D):
     Dual channel attention (AvgPool + MaxPool + Shared MLP) and 3D spatial depthwise attention gating to suppress
     surrounding abdominal background noise and focus on focal hyper/hypo-dense lesion clusters.
   - Edge-Aware Progressive Refinement Module 3D (PRM-3D):
     Progressively sharpens multi-scale predictions (16^3 -> 32^3 -> 64^3) with explicit 3D edge boundary tokens.

2. Co-Seg (MICCAI 2025):
   - Collaborative Learning with Mutual Prompt Guidance.
   - Liver is conditioned on Tumor prompt (z + g_tumor) to prevent cavity loss.
   - Tumor is conditioned on Liver prompt (z + g_liver) to strictly contain candidates within parenchyma.
"""

import os
import sys
import math
from typing import Tuple, Dict, Union, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.coseg_liver import ConvBlock3D, RPEncoder3D, MPDecoderHead3D


# ---------------------------------------------------------------------------
# 1. 3D Haar Discrete Wavelet Transform (DWT & IDWT)
# ---------------------------------------------------------------------------

class HaarDWT3D(nn.Module):
    """
    Computes 3D Discrete Wavelet Transform (Haar Wavelet) using grouped 3D convolutions.
    Decomposes (B, C, D, H, W) into:
    - 1 Low-Frequency subband (LLL): (B, C, D/2, H/2, W/2)
    - 7 High-Frequency subbands: (B, 7*C, D/2, H/2, W/2)
    """
    def __init__(self):
        super().__init__()
        l = torch.tensor([1.0, 1.0]) / (2.0 ** 0.5)
        h = torch.tensor([-1.0, 1.0]) / (2.0 ** 0.5)
        filters = []
        for i in [l, h]:
            for j in [l, h]:
                for k in [l, h]:
                    f = torch.einsum("i,j,k->ijk", i, j, k)
                    filters.append(f)
        filters = torch.stack(filters, dim=0).unsqueeze(1)  # (8, 1, 2, 2, 2)
        self.register_buffer("filters", filters.float())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, d, h, w = x.shape
        w_filt = self.filters.repeat(c, 1, 1, 1, 1)
        dwt_out = F.conv3d(x, w_filt, stride=2, groups=c)  # (B, 8*C, D/2, H/2, W/2)
        d2, h2, w2 = dwt_out.shape[-3:]
        dwt_out = dwt_out.view(b, c, 8, d2, h2, w2)
        lll = dwt_out[:, :, 0]
        high = dwt_out[:, :, 1:].reshape(b, c * 7, d2, h2, w2)
        return lll, high


class HaarIDWT3D(nn.Module):
    """
    Reconstructs spatial 3D feature map from LLL and 7 high-frequency wavelet subbands
    via transposed grouped 3D convolutions.
    """
    def __init__(self):
        super().__init__()
        l = torch.tensor([1.0, 1.0]) / (2.0 ** 0.5)
        h = torch.tensor([-1.0, 1.0]) / (2.0 ** 0.5)
        filters = []
        for i in [l, h]:
            for j in [l, h]:
                for k in [l, h]:
                    f = torch.einsum("i,j,k->ijk", i, j, k)
                    filters.append(f)
        filters = torch.stack(filters, dim=0).unsqueeze(1)  # (8, 1, 2, 2, 2)
        self.register_buffer("filters", filters.float())

    def forward(self, lll: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        b, c, d2, h2, w2 = lll.shape
        high_7 = high.view(b, c, 7, d2, h2, w2)
        lll_1 = lll.unsqueeze(2)
        x_dwt = torch.cat([lll_1, high_7], dim=2)  # (B, C, 8, D2, H2, W2)
        x_flat = x_dwt.view(b, c * 8, d2, h2, w2)
        w_filt = self.filters.repeat(c, 1, 1, 1, 1)
        out = F.conv_transpose3d(x_flat, w_filt, stride=2, groups=c)
        return out


# ---------------------------------------------------------------------------
# 2. 3D Global-Local Context Attention Module (GLCAM-3D)
# ---------------------------------------------------------------------------

class GLCAMContextAttention3D(nn.Module):
    """
    3D Global-Local Contextual Attention Module.
    Combines channel-wise attention with 3D spatial gating to suppress
    non-hepatic noise and spotlight small focal lesion clusters.
    """
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        red_dim = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv3d(in_channels, red_dim, kernel_size=1, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(red_dim, in_channels, kernel_size=1, bias=False)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.channel_mlp(self.avg_pool(x))
        max_out = self.channel_mlp(self.max_pool(x))
        ca = torch.sigmoid(avg_out + max_out)
        x_ca = x * ca

        # Spatial attention
        avg_spatial = torch.mean(x_ca, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_spatial, max_spatial], dim=1)
        sa = self.spatial_conv(spatial_cat)
        out = x_ca * sa

        return out + x  # Residual connection


# ---------------------------------------------------------------------------
# 3. 3D WaveFusion Refinement Block
# ---------------------------------------------------------------------------

class WaveFusionRefinementBlock3D(nn.Module):
    """
    3D WaveFusion Refinement Block (adapted from SAM2-CD / CanGio).
    Decomposes incoming features into high-frequency edge gradients and low-frequency semantics,
    processes each with dedicated residual pathways, and reconstructs via 3D IDWT.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dwt = HaarDWT3D()
        self.idwt = HaarIDWT3D()

        # High-frequency processing (3D Edge/boundary preservation)
        self.conv_high1 = nn.Conv3d(in_channels * 7, in_channels, kernel_size=1, bias=False)
        self.res_high = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels, affine=True)
        )
        self.conv_high2 = nn.Conv3d(in_channels, in_channels * 7, kernel_size=1, bias=False)

        # Low-frequency processing (Semantic shape & area)
        self.res_low = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels, affine=True)
        )

        # Output projection & residual connection
        self.out_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lll, high = self.dwt(x)

        # Process high-frequency subbands (edges)
        high_feat = self.conv_high1(high)
        high_res = self.res_high(high_feat)
        high_out = self.conv_high2(F.leaky_relu(high_feat + high_res, 0.01))

        # Process low-frequency subband (semantics)
        low_res = self.res_low(lll)
        low_out = F.leaky_relu(lll + low_res, 0.01)

        # Reconstruct via 3D IDWT
        rec = self.idwt(low_out, high_out)

        out = self.out_conv(rec) + self.shortcut(x)
        return out


# ---------------------------------------------------------------------------
# 4. WaveAttn 3D Multi-Scale Progressive Refinement Module (WaveAttn-PRM-3D)
# ---------------------------------------------------------------------------

class WaveAttnProgressiveRefinementModule3D(nn.Module):
    """
    Wavelet-Refined Progressive Refinement Module for 3D Medical CT.
    Operates across spatial scales:
      Scale 1/4 (16^3) -> Scale 1/2 (32^3) -> Scale 1/1 (64^3)
    Integrates:
    - GLCAM-3D attention on deep bottleneck and multi-scale skip features.
    - 3D WaveFusion blocks at each spatial scale.
    - Explicit multi-scale dual mask & edge boundary prediction [M, E].
    """
    def __init__(
        self,
        deep_dim: int = 192,
        skip_dims: List[int] = [96, 48, 24],  # S3 (16^3), S2 (32^3), S1 (64^3)
        feat_dim: int = 48,
        out_channels: int = 2  # 1 mask + 1 edge
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_channels = out_channels

        # GLCAM Context Attention Gates on deep and skip features
        self.glcam_deep = GLCAMContextAttention3D(deep_dim)
        self.glcam_s3 = GLCAMContextAttention3D(skip_dims[0])
        self.glcam_s2 = GLCAMContextAttention3D(skip_dims[1])
        self.glcam_s1 = GLCAMContextAttention3D(skip_dims[2])

        # Stage 1: 8^3 -> 16^3
        self.up_deep = nn.Sequential(
            nn.ConvTranspose3d(deep_dim, feat_dim, kernel_size=2, stride=2),
            nn.InstanceNorm3d(feat_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        # Input to Stage 1: [Fd_16; S3_16; Image_16; P0_16]
        in_ch1 = feat_dim + skip_dims[0] + 1 + out_channels
        self.fuse1 = nn.Sequential(
            nn.Conv3d(in_ch1, feat_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feat_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.wave1 = WaveFusionRefinementBlock3D(feat_dim, feat_dim)
        self.phi1 = nn.Conv3d(feat_dim, out_channels, kernel_size=1)

        # Stage 2: 16^3 -> 32^3
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(feat_dim, feat_dim, kernel_size=2, stride=2),
            nn.InstanceNorm3d(feat_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        # Input to Stage 2: [X1_32; S2_32; Image_32; P1_32]
        in_ch2 = feat_dim + skip_dims[1] + 1 + out_channels
        self.fuse2 = nn.Sequential(
            nn.Conv3d(in_ch2, feat_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feat_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.wave2 = WaveFusionRefinementBlock3D(feat_dim, feat_dim)
        self.phi2 = nn.Conv3d(feat_dim, out_channels, kernel_size=1)

        # Stage 3: 32^3 -> 64^3
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(feat_dim, feat_dim // 2, kernel_size=2, stride=2),
            nn.InstanceNorm3d(feat_dim // 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        # Input to Stage 3: [X2_64; S1_64; Image_64; P2_64]
        in_ch3 = (feat_dim // 2) + skip_dims[2] + 1 + out_channels
        self.fuse3 = nn.Sequential(
            nn.Conv3d(in_ch3, feat_dim // 2, kernel_size=1, bias=False),
            nn.InstanceNorm3d(feat_dim // 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.wave3 = WaveFusionRefinementBlock3D(feat_dim // 2, feat_dim // 2)
        self.phi3 = nn.Conv3d(feat_dim // 2, out_channels, kernel_size=1)

    def forward(
        self,
        deep_feats: torch.Tensor,
        skips: List[torch.Tensor],
        image_vol: torch.Tensor,
        init_preds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        deep_feats: (B, deep_dim, 8, 8, 8)
        skips: [S3 (16^3), S2 (32^3), S1 (64^3)]
        image_vol: (B, 1, 64, 64, 64)
        init_preds: (B, 2, 8, 8, 8) or (B, 2, 16, 16, 16) [M0; E0]
        """
        s3, s2, s1 = skips

        # 1. Apply GLCAM Context Attention
        deep_att = self.glcam_deep(deep_feats)
        s3_att = self.glcam_s3(s3)
        s2_att = self.glcam_s2(s2)
        s1_att = self.glcam_s1(s1)

        # --- Stage 1 (16^3) ---
        x1_in = self.up_deep(deep_att)
        i16 = F.interpolate(image_vol, size=(16, 16, 16), mode='trilinear', align_corners=False)
        p16 = F.interpolate(init_preds, size=(16, 16, 16), mode='trilinear', align_corners=False)
        cat1 = torch.cat([x1_in, s3_att, i16, p16], dim=1)
        feat1 = self.fuse1(cat1)
        x1 = self.wave1(feat1)
        p1 = self.phi1(x1)  # (B, 2, 16, 16, 16) [M1, E1]

        # --- Stage 2 (32^3) ---
        x2_in = self.up1(x1)
        i32 = F.interpolate(image_vol, size=(32, 32, 32), mode='trilinear', align_corners=False)
        p32 = F.interpolate(p1, size=(32, 32, 32), mode='trilinear', align_corners=False)
        cat2 = torch.cat([x2_in, s2_att, i32, p32], dim=1)
        feat2 = self.fuse2(cat2)
        x2 = self.wave2(feat2)
        p2 = self.phi2(x2)  # (B, 2, 32, 32, 32) [M2, E2]

        # --- Stage 3 (64^3) ---
        x3_in = self.up2(x2)
        p64 = F.interpolate(p2, size=(64, 64, 64), mode='trilinear', align_corners=False)
        cat3 = torch.cat([x3_in, s1_att, image_vol, p64], dim=1)
        feat3 = self.fuse3(cat3)
        x3 = self.wave3(feat3)
        p3 = self.phi3(x3)  # (B, 2, 64, 64, 64) [M3, E3]

        return {
            'p16': p1,
            'p32': p2,
            'p64': p3,
            'last_feat': x3
        }


# ---------------------------------------------------------------------------
# 5. Standalone 3D WaveAttn-SOPSeg Model
# ---------------------------------------------------------------------------

class WaveAttn_SOPSegLiverNet(nn.Module):
    """
    Standalone 3D WaveAttn-SOPSeg Multi-Task Network for Abdominal CT.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 6,
        num_cls_classes: int = 5,
        base_channels: int = 24
    ):
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes

        c1 = base_channels       # 24
        c2 = base_channels * 2   # 48
        c3 = base_channels * 4   # 96
        c4 = base_channels * 8   # 192

        # 3D Feature Extractor
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c1, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.stage1 = ConvBlock3D(c1, c1, stride=1)  # 64^3
        self.stage2 = ConvBlock3D(c1, c2, stride=2)  # 32^3
        self.stage3 = ConvBlock3D(c2, c3, stride=2)  # 16^3
        self.stage4 = ConvBlock3D(c3, c4, stride=2)  # 8^3

        # Coarse Initial Predictors [Mask0, Edge0]
        self.coarse_liver_head = nn.Conv3d(c4, 2, kernel_size=1)
        self.coarse_tumor_head = nn.Conv3d(c4, 2, kernel_size=1)

        # Dual WaveAttn Multi-Scale PRM Modules
        self.prm_liver = WaveAttnProgressiveRefinementModule3D(
            deep_dim=c4,
            skip_dims=[c3, c2, c1],
            feat_dim=48,
            out_channels=2
        )
        self.prm_tumor = WaveAttnProgressiveRefinementModule3D(
            deep_dim=c4,
            skip_dims=[c3, c2, c1],
            feat_dim=48,
            out_channels=2
        )

        # Subtype Classifier Head
        self.subtype_head = nn.Conv3d(24, 4, kernel_size=1)

        # Multi-task Global Heads
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(c4, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_cls_classes)
        )
        self.burden_head = nn.Sequential(
            nn.Linear(c4, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Feature extraction
        x0 = self.stem(x)
        s1 = self.stage1(x0)  # 64^3
        s2 = self.stage2(s1)  # 32^3
        s3 = self.stage3(s2)  # 16^3
        z = self.stage4(s3)   # 8^3

        # Initial coarse predictions [M0, E0]
        init_liver = self.coarse_liver_head(z)
        init_tumor = self.coarse_tumor_head(z)

        # Progressive WaveFusion Refinement
        ref_liver = self.prm_liver(z, [s3, s2, s1], x, init_liver)
        ref_tumor = self.prm_tumor(z, [s3, s2, s1], x, init_tumor)

        # Final predictions (64^3)
        liv_p64 = ref_liver['p64']
        tum_p64 = ref_tumor['p64']

        pred_liver_mask = liv_p64[:, 0:1]
        pred_liver_edge = liv_p64[:, 1:2]
        pred_tumor_mask = tum_p64[:, 0:1]
        pred_tumor_edge = tum_p64[:, 1:2]

        subtypes = self.subtype_head(ref_tumor['last_feat'])

        # Unified 6-class segmentation
        bg_logit = -pred_liver_mask
        liver_parenchyma = pred_liver_mask - pred_tumor_mask
        lesion_logits = subtypes + pred_tumor_mask
        full_seg = torch.cat([bg_logit, liver_parenchyma, lesion_logits], dim=1)

        # Global multi-task heads
        z_pool = self.pool(z).flatten(1)
        cls_logits = self.cls_head(z_pool)
        tumor_burden = self.burden_head(z_pool)

        # Interpolate coarse for prior loss if needed
        m0_liv_64 = F.interpolate(init_liver[:, 0:1], size=(64, 64, 64), mode='trilinear', align_corners=False)
        m0_tum_64 = F.interpolate(init_tumor[:, 0:1], size=(64, 64, 64), mode='trilinear', align_corners=False)

        return {
            'seg_logits': full_seg,
            'refined_liver_mask': pred_liver_mask,
            'refined_liver_edge': pred_liver_edge,
            'refined_tumor_mask': pred_tumor_mask,
            'refined_tumor_edge': pred_tumor_edge,
            'pred_liver_mask': pred_liver_mask,
            'pred_liver_edge': pred_liver_edge,
            'pred_tumor_mask': pred_tumor_mask,
            'pred_tumor_edge': pred_tumor_edge,
            'coarse_liver_logits': m0_liv_64,
            'coarse_tumor_logits': m0_tum_64,
            'p16_liver': ref_liver['p16'],
            'p32_liver': ref_liver['p32'],
            'p16_tumor': ref_tumor['p16'],
            'p32_tumor': ref_tumor['p32'],
            'cls_logits': cls_logits,
            'tumor_burden': tumor_burden
        }


# ---------------------------------------------------------------------------
# 6. Combined Co-Seg + WaveAttn-SOPSeg Collaborative Network
# ---------------------------------------------------------------------------

class CoSeg_WaveAttn_SOPSegLiverNet(nn.Module):
    """
    Combined Co-Seg + WaveAttn-SOPSeg Collaborative Architecture for LiMT.

    Synergistic Triad:
    1. Co-Seg Mutual Cross-Guidance:
       Liver prompt (g_liver) guides tumor decoder, eliminating spurious out-of-liver predictions.
       Tumor prompt (g_tumor) guides liver decoder, preserving necrotic/hypodense boundaries.
    2. WaveAttn 3D WaveFusion:
       HaarDWT3D isolates high-frequency surface gradients (LH, HL, HH, etc.) from low-frequency
       organ density (LLL), preventing fine lesion blurring across multi-scale upsampling.
    3. GLCAM-3D Context Attention:
       Dynamic channel + spatial attention gating spotlights tiny lesion clusters against
       parenchyma variations.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 6,
        num_cls_classes: int = 5,
        base_channels: int = 24
    ):
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes

        c1 = base_channels       # 24
        c2 = base_channels * 2   # 48
        c3 = base_channels * 4   # 96
        c4 = base_channels * 8   # 192
        self.embed_dim = c4

        # Shared 3D Backbone
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c1, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.stage1 = ConvBlock3D(c1, c1, stride=1)  # 64^3
        self.stage2 = ConvBlock3D(c1, c2, stride=2)  # 32^3
        self.stage3 = ConvBlock3D(c2, c3, stride=2)  # 16^3
        self.stage4 = ConvBlock3D(c3, c4, stride=2)  # 8^3

        # Forward 1 Coarse Heads: predicts initial mask and edge [M0, E0]
        self.coarse_liver_head = nn.Conv3d(c4, 2, kernel_size=1)
        self.coarse_tumor_head = nn.Conv3d(c4, 2, kernel_size=1)

        # Co-Seg 3D RP-Encoders (takes binary/coarse mask logits)
        self.rp_encoder_liver = RPEncoder3D(in_channels=1, embed_dim=c4, num_heads=4)
        self.rp_encoder_tumor = RPEncoder3D(in_channels=1, embed_dim=c4, num_heads=4)

        # Co-Seg Mutual Query Interaction Layers
        self.query_liver = nn.Parameter(torch.randn(1, 16, c4) * 0.02)
        self.query_tumor = nn.Parameter(torch.randn(1, 16, c4) * 0.02)

        self.attn_liver_sa = nn.MultiheadAttention(c4, num_heads=4, batch_first=True)
        self.attn_liver_ca = nn.MultiheadAttention(c4, num_heads=4, batch_first=True)
        self.norm_liv_q = nn.LayerNorm(c4)
        self.attn_liver_rev = nn.MultiheadAttention(c4, num_heads=4, batch_first=True)
        self.norm_liv_feat = nn.LayerNorm(c4)

        self.attn_tumor_sa = nn.MultiheadAttention(c4, num_heads=4, batch_first=True)
        self.attn_tumor_ca = nn.MultiheadAttention(c4, num_heads=4, batch_first=True)
        self.norm_tum_q = nn.LayerNorm(c4)
        self.attn_tumor_rev = nn.MultiheadAttention(c4, num_heads=4, batch_first=True)
        self.norm_tum_feat = nn.LayerNorm(c4)

        # WaveAttn Progressive Refinement Modules (WaveAttn-PRM-3D)
        self.prm_liver = WaveAttnProgressiveRefinementModule3D(
            deep_dim=c4,
            skip_dims=[c3, c2, c1],
            feat_dim=48,
            out_channels=2
        )
        self.prm_tumor = WaveAttnProgressiveRefinementModule3D(
            deep_dim=c4,
            skip_dims=[c3, c2, c1],
            feat_dim=48,
            out_channels=2
        )

        # Subtype Classifier Head (last_feat has feat_dim // 2 = 24 channels)
        self.subtype_head = nn.Conv3d(24, 4, kernel_size=1)

        # Multi-task Global Heads
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(c4, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_cls_classes)
        )
        self.burden_head = nn.Sequential(
            nn.Linear(c4, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = x.shape[0]

        # Stage A: 3D Multi-scale Backbone Extraction
        x0 = self.stem(x)
        s1 = self.stage1(x0)  # 64^3
        s2 = self.stage2(s1)  # 32^3
        s3 = self.stage3(s2)  # 16^3
        z = self.stage4(s3)   # 8^3, (B, C4, 8, 8, 8)

        # Stage B: Forward 1 Coarse Multi-Task Predictions
        init_liver = self.coarse_liver_head(z)  # (B, 2, 8, 8, 8) [M0, E0]
        init_tumor = self.coarse_tumor_head(z)  # (B, 2, 8, 8, 8) [M0, E0]

        coarse_m_liver = init_liver[:, 0:1]
        coarse_m_tumor = init_tumor[:, 0:1]

        # Upsample coarse masks to 64^3 for RP-Encoder and prior loss
        m0_liv_64 = F.interpolate(coarse_m_liver, size=(64, 64, 64), mode='trilinear', align_corners=False)
        m0_tum_64 = F.interpolate(coarse_m_tumor, size=(64, 64, 64), mode='trilinear', align_corners=False)

        # Stage C: Co-Seg 3D RP-Encoders (Extract Region Prompts)
        g_liver = self.rp_encoder_liver(torch.sigmoid(m0_liv_64), z)  # (B, C4, 8, 8, 8)
        g_tumor = self.rp_encoder_tumor(torch.sigmoid(m0_tum_64), z)  # (B, C4, 8, 8, 8)

        # Stage D: Co-Seg Mutual Query Cross-Guidance
        z_liv_guided = z + g_tumor  # Liver conditioned on tumor prompt
        z_tum_guided = z + g_liver  # Tumor conditioned on liver prompt

        # Flatten for attention
        z_liv_seq = z_liv_guided.flatten(2).transpose(1, 2)  # (B, 512, C4)
        z_tum_seq = z_tum_guided.flatten(2).transpose(1, 2)  # (B, 512, C4)

        # Liver queries
        q_liv = self.query_liver.repeat(b, 1, 1)
        q_liv_sa, _ = self.attn_liver_sa(q_liv, q_liv, q_liv)
        q_liv = self.norm_liv_q(q_liv + q_liv_sa)
        q_liv_ca, _ = self.attn_liver_ca(q_liv, z_liv_seq, z_liv_seq)
        q_liv = self.norm_liv_q(q_liv + q_liv_ca)
        feat_liv_seq, _ = self.attn_liver_rev(z_liv_seq, q_liv, q_liv)
        z_liv_refined = self.norm_liv_feat(z_liv_seq + feat_liv_seq).transpose(1, 2).view_as(z)

        # Tumor queries
        q_tum = self.query_tumor.repeat(b, 1, 1)
        q_tum_sa, _ = self.attn_tumor_sa(q_tum, q_tum, q_tum)
        q_tum = self.norm_tum_q(q_tum + q_tum_sa)
        q_tum_ca, _ = self.attn_tumor_ca(q_tum, z_tum_seq, z_tum_seq)
        q_tum = self.norm_tum_q(q_tum + q_tum_ca)
        feat_tum_seq, _ = self.attn_tumor_rev(z_tum_seq, q_tum, q_tum)
        z_tum_refined = self.norm_tum_feat(z_tum_seq + feat_tum_seq).transpose(1, 2).view_as(z)

        # Stage E: WaveAttn 3D Multi-Scale Progressive Refinement (PRM-3D)
        ref_liver = self.prm_liver(z_liv_refined, [s3, s2, s1], x, init_liver)
        ref_tumor = self.prm_tumor(z_tum_refined, [s3, s2, s1], x, init_tumor)

        # Final predictions (64^3)
        liv_p64 = ref_liver['p64']
        tum_p64 = ref_tumor['p64']

        pred_liver_mask = liv_p64[:, 0:1]
        pred_liver_edge = liv_p64[:, 1:2]
        pred_tumor_mask = tum_p64[:, 0:1]
        pred_tumor_edge = tum_p64[:, 1:2]

        # Lesion Subtype Classifier Head
        subtypes = self.subtype_head(ref_tumor['last_feat'])

        # Unified 6-class segmentation
        bg_logit = -pred_liver_mask
        liver_parenchyma = pred_liver_mask - pred_tumor_mask
        lesion_logits = subtypes + pred_tumor_mask
        full_seg = torch.cat([bg_logit, liver_parenchyma, lesion_logits], dim=1)

        # Global multi-task heads
        z_pool = self.pool(z).flatten(1)
        cls_logits = self.cls_head(z_pool)
        tumor_burden = self.burden_head(z_pool)

        return {
            'seg_logits': full_seg,
            'refined_liver_mask': pred_liver_mask,
            'refined_liver_edge': pred_liver_edge,
            'refined_tumor_mask': pred_tumor_mask,
            'refined_tumor_edge': pred_tumor_edge,
            'pred_liver_mask': pred_liver_mask,
            'pred_liver_edge': pred_liver_edge,
            'pred_tumor_mask': pred_tumor_mask,
            'pred_tumor_edge': pred_tumor_edge,
            'coarse_liver_logits': m0_liv_64,
            'coarse_tumor_logits': m0_tum_64,
            'p16_liver': ref_liver['p16'],
            'p32_liver': ref_liver['p32'],
            'p16_tumor': ref_tumor['p16'],
            'p32_tumor': ref_tumor['p32'],
            'cls_logits': cls_logits,
            'tumor_burden': tumor_burden
        }
