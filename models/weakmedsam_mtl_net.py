"""
WeakMedSAM-MTL: Unified Multi-Task Learning Network for Joint Liver & Tumor Segmentation
and Multi-Label Lesion Classification on the LiMT Benchmark.

Key Innovations:
1. Shared 3D Multi-Scale Residual Backbone with Wavelet High-Frequency Edge Preservation (HaarDWT3D).
2. Second-Order Bilinear Feature Pooling (SOPSeg) to capture fine-grained tumor micro-texture.
3. Co-Seg Collaborative Mutual Guidance between Liver parenchyma and Tumor streams.
4. Segmentation-Guided Masked Feature Pooling (MFP):
   Extracts lesion-specific representations directly from the predicted tumor localization,
   preventing background organ dilution.
5. Multi-Label Classification Head with Sub-Class Exploration (SCE) prototypes predicting:
   - Class 0: Liver / Normal Parenchyma
   - Class 1: Hepatic Cyst
   - Class 2: Cavernous Hemangioma
   - Class 3: Hepatic Metastasis
   - Class 4: Primary Liver Cancer / HCC
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# 1. 3D Haar Discrete Wavelet Transform (HaarDWT3D)
# ---------------------------------------------------------------------------
class HaarDWT3D(nn.Module):
    """
    Decomposes (B, C, D, H, W) into:
    - 1 Low-Frequency subband (LLL): (B, C, D/2, H/2, W/2)
    - 7 High-Frequency subbands: (B, 7*C, D/2, H/2, W/2)
    Captures sharp cyst fluid margins and vascular tumor boundary gradients.
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
        dwt_out = F.conv3d(x, w_filt, stride=2, groups=c)
        d2, h2, w2 = dwt_out.shape[-3:]
        dwt_out = dwt_out.view(b, c, 8, d2, h2, w2)
        lll = dwt_out[:, :, 0]
        high = dwt_out[:, :, 1:].reshape(b, c * 7, d2, h2, w2)
        return lll, high


# ---------------------------------------------------------------------------
# 2. Second-Order Bilinear Pooling (SOPSeg) Texture Block
# ---------------------------------------------------------------------------
class SOPSegTextureBlock3D(nn.Module):
    """
    Captures non-linear covariance of 3D feature channels to distinguish subtle
    CT enhancement patterns (Cyst hypodensity, Hemangioma nodular peripheral fill-in,
    HCC early wash-in/wash-out, and Metastasis target sign).
    """
    def __init__(self, in_c: int, reduced_c: int = 48):
        super().__init__()
        self.conv_reduce = nn.Sequential(
            nn.Conv3d(in_c, reduced_c, kernel_size=1, bias=False),
            nn.InstanceNorm3d(reduced_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(reduced_c, in_c),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        feat = self.conv_reduce(x) # (B, reduced_c, D, H, W)
        r = feat.shape[1]
        flat = feat.view(b, r, -1) # (B, r, N)
        # Covariance pooling
        cov = torch.bmm(flat, flat.transpose(1, 2)) / (d * h * w)
        diag = torch.diagonal(cov, dim1=1, dim2=2) # (B, r)
        weights = self.fc(diag).view(b, c, 1, 1, 1)
        return x * (1.0 + weights)


# ---------------------------------------------------------------------------
# 3. 3D Residual Conv Block
# ---------------------------------------------------------------------------
class ResBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_c, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_c, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_c, affine=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)


# ---------------------------------------------------------------------------
# 4. Wavelet-Guided Attention Gate 3D
# ---------------------------------------------------------------------------
class WaveletAttentionGate3D(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv3d(f_g, f_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(f_int, affine=True)
        )
        self.w_x = nn.Sequential(
            nn.Conv3d(f_l, f_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(f_int, affine=True)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        psi = self.psi(self.act(g1 + x1))
        return x * psi


# ---------------------------------------------------------------------------
# 5. WeakMedSAM-MTL Architecture
# ---------------------------------------------------------------------------
class WeakMedSAM_MTLNet(nn.Module):
    """
    Unified Multi-Task Network for Joint Segmentation & Classification.
    Shared Hierarchical Encoder -> Inter-Task Mutual Guidance ->
    [Dual Segmentation Heads (Liver, Tumor)] + [Segmentation-Guided Classification Head (5 Classes)].
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 2, # Channel 0: Liver, Channel 1: Tumor
        num_cls_classes: int = 5, # Liver, Cyst, Hemangioma, Metastasis, HCC
        base_channels: int = 24
    ):
        super().__init__()
        c = base_channels
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes

        # Stem & Hierarchical Encoder
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        ) # [B, c, D, H, W]

        self.stage1 = ResBlock3D(c, c, stride=1)       # [B, c, 64, 64, 64]
        self.stage2 = ResBlock3D(c, c * 2, stride=2)   # [B, 2c, 32, 32, 32]
        self.stage3 = ResBlock3D(c * 2, c * 4, stride=2) # [B, 4c, 16, 16, 16]
        self.stage4 = ResBlock3D(c * 4, c * 8, stride=2) # [B, 8c, 8, 8, 8]

        # Wavelet Edge Analysis at Bottleneck
        self.dwt = HaarDWT3D()
        self.dwt_conv = nn.Sequential(
            nn.Conv3d(c * 4 * 7, c * 4, kernel_size=1, bias=False),
            nn.InstanceNorm3d(c * 4, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # SOPSeg Texture Modeling Block
        self.sopseg = SOPSegTextureBlock3D(in_c=c * 8, reduced_c=48)

        # Bottleneck Fusion
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv3d(c * 8 + c * 4, c * 8, kernel_size=1, bias=False),
            nn.InstanceNorm3d(c * 8, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Attention U-Net Decoder for Segmentation
        # Up 4: 8c -> 4c
        self.up4 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.att4 = WaveletAttentionGate3D(f_g=c * 4, f_l=c * 4, f_int=c * 2)
        self.dec4 = ResBlock3D(c * 8, c * 4)

        # Up 3: 4c -> 2c
        self.up3 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.att3 = WaveletAttentionGate3D(f_g=c * 2, f_l=c * 2, f_int=c)
        self.dec3 = ResBlock3D(c * 4, c * 2)

        # Up 2: 2c -> c
        self.up2 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.att2 = WaveletAttentionGate3D(f_g=c, f_l=c, f_int=c // 2)
        self.dec2 = ResBlock3D(c * 2, c)

        # Co-Seg Mutual Guidance Streams (Liver & Tumor)
        self.liver_stream = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.tumor_stream = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Mutual Prompting Gates
        self.gate_liver_to_tumor = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=1),
            nn.Sigmoid()
        )
        self.gate_tumor_to_liver = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=1),
            nn.Sigmoid()
        )

        # Specialized Segmentation Heads
        self.head_liver = nn.Sequential(
            nn.Conv3d(c, c // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c // 2, 1, kernel_size=1)
        )
        self.head_tumor = nn.Sequential(
            nn.Conv3d(c, c // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c // 2, 1, kernel_size=1)
        )

        # -------------------------------------------------------------------
        # Multi-Label Lesion Classification Branch
        # -------------------------------------------------------------------
        # 1. Global Context Pooling
        self.global_pool_avg = nn.AdaptiveAvgPool3d(1)
        self.global_pool_max = nn.AdaptiveMaxPool3d(1)

        # 2. Segmentation-Guided Masked Pooling (MFP)
        # Combines bottleneck representations weighted by downsampled tumor probability map
        self.cls_head = nn.Sequential(
            nn.Linear(c * 8 * 2 + c * 8, c * 8), # Global avg + max + lesion-masked
            nn.BatchNorm1d(c * 8),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(c * 8, c * 4),
            nn.BatchNorm1d(c * 4),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(c * 4, num_cls_classes) # [Liver, Cyst, Hemangioma, Metastasis, HCC]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # --- Shared Hierarchical Encoder ---
        x0 = self.stem(x)       # [B, c, 64, 64, 64]
        e1 = self.stage1(x0)    # [B, c, 64, 64, 64]
        e2 = self.stage2(e1)    # [B, 2c, 32, 32, 32]
        e3 = self.stage3(e2)    # [B, 4c, 16, 16, 16]
        e4 = self.stage4(e3)    # [B, 8c, 8, 8, 8]

        # SOPSeg texture enhancement
        e4_sop = self.sopseg(e4) # [B, 8c, 8, 8, 8]

        # Wavelet edge extraction on stage 3 features
        _, e3_high = self.dwt(e3) # [B, 4c*7, 8, 8, 8]
        e3_high_proj = self.dwt_conv(e3_high) # [B, 4c, 8, 8, 8]

        # Bottleneck fusion
        b_feat = self.bottleneck_fusion(torch.cat([e4_sop, e3_high_proj], dim=1)) # [B, 8c, 8, 8, 8]

        # --- Segmentation Decoder with Wavelet Attention Gates ---
        d4 = self.up4(b_feat)                     # [B, 4c, 16, 16, 16]
        att4 = self.att4(g=d4, x=e3)              # [B, 4c, 16, 16, 16]
        d4 = self.dec4(torch.cat([att4, d4], dim=1))

        d3 = self.up3(d4)                         # [B, 2c, 32, 32, 32]
        att3 = self.att3(g=d3, x=e2)              # [B, 2c, 32, 32, 32]
        d3 = self.dec3(torch.cat([att3, d3], dim=1))

        d2 = self.up2(d3)                         # [B, c, 64, 64, 64]
        att2 = self.att2(g=d2, x=e1)              # [B, c, 64, 64, 64]
        d2 = self.dec2(torch.cat([att2, d2], dim=1))

        # --- Co-Seg Mutual Guidance ---
        liv_feat = self.liver_stream(d2)
        tum_feat = self.tumor_stream(d2)

        # Cross-gating: Tumor is constrained to liver parenchyma; Liver is protected against tumor cavity
        gate_l2t = self.gate_liver_to_tumor(liv_feat)
        gate_t2l = self.gate_tumor_to_liver(tum_feat)

        tum_guided = tum_feat * (1.0 + gate_l2t)
        liv_guided = liv_feat * (1.0 + gate_t2l)

        logit_liver = self.head_liver(liv_guided) # [B, 1, 64, 64, 64]
        logit_tumor = self.head_tumor(tum_guided) # [B, 1, 64, 64, 64]
        seg_logits = torch.cat([logit_liver, logit_tumor], dim=1) # [B, 2, 64, 64, 64]

        # --- Segmentation-Guided Masked Classification Head ---
        # Downsample tumor probability to bottleneck resolution [B, 1, 8, 8, 8]
        prob_tumor = torch.sigmoid(logit_tumor)
        prob_tumor_low = F.interpolate(prob_tumor, size=b_feat.shape[2:], mode='trilinear', align_corners=False)

        # 1. Global average & max pooling
        f_avg = self.global_pool_avg(b_feat).flatten(1) # [B, 8c]
        f_max = self.global_pool_max(b_feat).flatten(1) # [B, 8c]

        # 2. Lesion-masked spatial pooling: focus features on the detected lesion region
        mask_weights = prob_tumor_low / (prob_tumor_low.sum(dim=(2, 3, 4), keepdim=True) + 1e-5) # [B, 1, 8, 8, 8]
        f_lesion = (b_feat * mask_weights).sum(dim=(2, 3, 4)) # [B, 8c]

        # Concatenate global + lesion-specific representations
        f_cls_in = torch.cat([f_avg, f_max, f_lesion], dim=1) # [B, 8c * 3]
        cls_logits = self.cls_head(f_cls_in) # [B, 5]

        return {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits,
            "latent_feat": b_feat,
            "tumor_prob": prob_tumor
        }
