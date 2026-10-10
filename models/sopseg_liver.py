"""
SOPSeg & Co-Seg Combined Architecture for 3D Volumetric Liver & Tumor CT Multi-Task Learning.

Adapted from:
1. SOPSeg (from cangio project: `cangio/src/sopseg.py`, Wang et al. AAAI 2026 / 2025):
   - Small Object Instance Segmentation
   - Edge-Aware Decoder with Explicit Boundary Tokens
   - Multi-Scale Progressive Refinement Module (PRM): 1/8 -> 1/4 -> 1/2 -> 1/1 with Residual Refinement Blocks (RRB)
2. Co-Seg (MICCAI 2025):
   - Collaborative Learning with Mutual Prompt Guidance
   - 3D Region-Aware Prompt Encoder (RP-Encoder)
   - 3D Mutual Prompt Mask Decoder (MP-Decoder)

Synergy of Co-Seg + SOPSeg:
- Co-Seg provides MACRO-MICRO MUTUAL GUIDANCE:
  Liver prompt constrains tumor candidates strictly within parenchyma, suppressing out-of-liver false positives.
  Tumor prompt informs liver boundary to prevent necrotic/hypodense cavity under-segmentation.
- SOPSeg provides SMALL-OBJECT PROGRESSIVE EDGE REFINEMENT:
  Tumors in LiMT are small objects (1-3% of liver volume).
  PRM-3D fuses shallow spatial cues, image intensity guidance, and explicit edge boundary predictions
  to progressively sharpen tumor contours across 16^3 -> 32^3 -> 64^3 scales.
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


class ResidualRefinementBlock3D(nn.Module):
    """
    3D Residual Refinement Block (RRB-3D) adapted from SOPSeg.
    Fuses previous scale representations, high-res CT intensity guidance,
    and intermediate mask/edge predictions.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True)
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
            if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv2(self.conv1(x))
        return self.act(out + res)


class ProgressiveRefinementModule3D(nn.Module):
    """
    3D Multi-Scale Progressive Refinement Module (PRM-3D) adapted from SOPSeg.
    Operates across spatial scales:
      Scale 1/4 (16x16x16) -> Scale 1/2 (32x32x32) -> Scale 1/1 (64x64x64)
    At each scale, outputs mask predictions M and edge boundary predictions E: [M, E].
    """
    def __init__(
        self,
        deep_dim: int = 192,
        skip_dims: List[int] = [96, 48, 24], # S3 (16^3), S2 (32^3), S1 (64^3)
        feat_dim: int = 48,
        out_channels: int = 2 # 1 mask + 1 edge
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_channels = out_channels
        
        # Stage 1: 8x8x8 -> 16x16x16
        self.up_deep = nn.Sequential(
            nn.ConvTranspose3d(deep_dim, feat_dim, kernel_size=2, stride=2),
            nn.InstanceNorm3d(feat_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        # Input: [Fd_16; S3_16; Image_16; P0_16]
        in_ch1 = feat_dim + skip_dims[0] + 1 + out_channels
        self.r1 = ResidualRefinementBlock3D(in_ch1, feat_dim)
        self.phi1 = nn.Conv3d(feat_dim, out_channels, kernel_size=1)
        
        # Stage 2: 16x16x16 -> 32x32x32
        self.up1 = nn.ConvTranspose3d(feat_dim, feat_dim, kernel_size=2, stride=2)
        # Input: [X1; S2_32; Image_32; P1_32]
        in_ch2 = feat_dim + skip_dims[1] + 1 + out_channels
        self.r2 = ResidualRefinementBlock3D(in_ch2, feat_dim)
        self.phi2 = nn.Conv3d(feat_dim, out_channels, kernel_size=1)
        
        # Stage 3: 32x32x32 -> 64x64x64
        self.up2 = nn.ConvTranspose3d(feat_dim, feat_dim // 2, kernel_size=2, stride=2)
        # Input: [X2; S1_64; Image_64; P2_64]
        in_ch3 = (feat_dim // 2) + skip_dims[2] + 1 + out_channels
        self.r3 = ResidualRefinementBlock3D(in_ch3, feat_dim // 2)
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
        
        # --- Stage 1 (16^3) ---
        x1_in = self.up_deep(deep_feats) # (B, feat_dim, 16, 16, 16)
        i16 = F.interpolate(image_vol, size=(16, 16, 16), mode='trilinear', align_corners=False)
        p16 = F.interpolate(init_preds, size=(16, 16, 16), mode='trilinear', align_corners=False)
        cat1 = torch.cat([x1_in, s3, i16, p16], dim=1)
        x1 = self.r1(cat1)
        p1 = self.phi1(x1) # (B, 2, 16, 16, 16) [M1, E1]
        
        # --- Stage 2 (32^3) ---
        x2_in = self.up1(x1) # (B, feat_dim, 32, 32, 32)
        i32 = F.interpolate(image_vol, size=(32, 32, 32), mode='trilinear', align_corners=False)
        p32 = F.interpolate(p1, size=(32, 32, 32), mode='trilinear', align_corners=False)
        cat2 = torch.cat([x2_in, s2, i32, p32], dim=1)
        x2 = self.r2(cat2)
        p2 = self.phi2(x2) # (B, 2, 32, 32, 32) [M2, E2]
        
        # --- Stage 3 (64^3) ---
        x3_in = self.up2(x2) # (B, feat_dim // 2, 64, 64, 64)
        p64 = F.interpolate(p2, size=(64, 64, 64), mode='trilinear', align_corners=False)
        cat3 = torch.cat([x3_in, s1, image_vol, p64], dim=1)
        x3 = self.r3(cat3)
        p3 = self.phi3(x3) # (B, 2, 64, 64, 64) [M3, E3]
        
        return {
            'p16': p1, # 1/4 scale: [M_16, E_16]
            'p32': p2, # 1/2 scale: [M_32, E_32]
            'p64': p3, # 1/1 scale: [M_64, E_64]
            'last_feat': x3
        }


class SOPSegLiverNet(nn.Module):
    """
    SOPSeg 3D Volumetric Model for LiMT Liver & Tumor Segmentation.
    Implements SOPSeg's Edge-Aware Multi-Scale Progressive Refinement Architecture.
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
        
        # 3D Feature Backbone
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c1, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.stage1 = ConvBlock3D(c1, c1, stride=1) # 64x64x64
        self.stage2 = ConvBlock3D(c1, c2, stride=2) # 32x32x32
        self.stage3 = ConvBlock3D(c2, c3, stride=2) # 16x16x16
        self.stage4 = ConvBlock3D(c3, c4, stride=2) # 8x8x8
        
        # Initial Coarse Predictor (P0 = [M0, E0] for both liver and tumor)
        self.coarse_head = nn.Sequential(
            nn.Conv3d(c4, c2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c2, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c2, 4, kernel_size=1) # [M_liv0, E_liv0, M_tum0, E_tum0]
        )
        
        # Progressive Refinement Module for Liver (M_liv, E_liv)
        self.prm_liver = ProgressiveRefinementModule3D(
            deep_dim=c4,
            skip_dims=[c3, c2, c1],
            feat_dim=48,
            out_channels=2
        )
        
        # Progressive Refinement Module for Tumor (M_tum, E_tum)
        self.prm_tumor = ProgressiveRefinementModule3D(
            deep_dim=c4,
            skip_dims=[c3, c2, c1],
            feat_dim=48,
            out_channels=2
        )
        
        # Subtype classifier on refined features (last_feat has feat_dim // 2 = 24 channels)
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
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Backbone
        s1 = self.stage1(self.stem(x))      # 64^3
        s2 = self.stage2(s1)                 # 32^3
        s3 = self.stage3(s2)                 # 16^3
        z = self.stage4(s3)                  # 8^3
        skips = [s3, s2, s1]
        
        # Coarse Stage
        p0 = self.coarse_head(z)             # (B, 4, 8, 8, 8)
        p0_liver = p0[:, 0:2]                # [M0, E0] liver
        p0_tumor = p0[:, 2:4]                # [M0, E0] tumor
        
        # Progressive Refinements
        ref_liver = self.prm_liver(z, skips, x, p0_liver)
        ref_tumor = self.prm_tumor(z, skips, x, p0_tumor)
        
        # Final Full-Resolution Predictions (64^3)
        liv_p64 = ref_liver['p64']
        tum_p64 = ref_tumor['p64']
        
        pred_liver_mask = liv_p64[:, 0:1]
        pred_liver_edge = liv_p64[:, 1:2]
        pred_tumor_mask = tum_p64[:, 0:1]
        pred_tumor_edge = tum_p64[:, 1:2]
        
        # Subtypes
        subtypes = self.subtype_head(ref_tumor['last_feat'])
        
        # Unified 6-class segmentation
        bg_logit = -pred_liver_mask
        liver_parenchyma = pred_liver_mask - pred_tumor_mask
        lesion_logits = subtypes + pred_tumor_mask
        full_seg = torch.cat([bg_logit, liver_parenchyma, lesion_logits], dim=1)
        
        # Multi-task heads
        z_pool = self.pool(z).flatten(1)
        cls_logits = self.cls_head(z_pool)
        tumor_burden = self.burden_head(z_pool)
        
        return {
            'seg_logits': full_seg,
            'pred_liver_mask': pred_liver_mask,
            'pred_liver_edge': pred_liver_edge,
            'pred_tumor_mask': pred_tumor_mask,
            'pred_tumor_edge': pred_tumor_edge,
            'p16_liver': ref_liver['p16'],
            'p32_liver': ref_liver['p32'],
            'p16_tumor': ref_tumor['p16'],
            'p32_tumor': ref_tumor['p32'],
            'cls_logits': cls_logits,
            'tumor_burden': tumor_burden
        }


class CoSeg_SOPSeg_LiverNet(nn.Module):
    """
    Combined Co-Seg + SOPSeg Collaborative Network for LiMT.
    
    Combines:
    1. Co-Seg 3D RP-Encoder & MP-Decoder:
       Bidirectional mutual prompt guidance between Liver (organ) and Tumor (lesions).
       Liver is conditioned on Tumor prompt (z + g_tumor).
       Tumor is conditioned on Liver prompt (z + g_liver).
    2. SOPSeg 3D Progressive Refinement Module (PRM-3D):
       Takes the mutually guided representations and applies edge-aware multi-scale
       progressive residual refinement across 16^3 -> 32^3 -> 64^3 scales.
    3. Multi-Task Global Heads:
       Multi-label classification and tumor burden regression.
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
        self.stage1 = ConvBlock3D(c1, c1, stride=1) # 64x64x64
        self.stage2 = ConvBlock3D(c1, c2, stride=2) # 32x32x32
        self.stage3 = ConvBlock3D(c2, c3, stride=2) # 16x16x16
        self.stage4 = ConvBlock3D(c3, c4, stride=2) # 8x8x8
        
        # Forward 1 Coarse Heads: predicts initial mask and edge [M0, E0]
        self.coarse_liver_head = nn.Conv3d(c4, 2, kernel_size=1) # [M0, E0]
        self.coarse_tumor_head = nn.Conv3d(c4, 2, kernel_size=1) # [M0, E0]
        
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
        
        # SOPSeg Progressive Refinement Modules (PRM-3D)
        # Refines Liver with mutual tumor prompt guidance
        self.prm_liver = ProgressiveRefinementModule3D(
            deep_dim=c4,
            skip_dims=[c3, c2, c1],
            feat_dim=48,
            out_channels=2 # [Mask, Edge]
        )
        
        # Refines Tumor with mutual liver prompt guidance
        self.prm_tumor = ProgressiveRefinementModule3D(
            deep_dim=c4,
            skip_dims=[c3, c2, c1],
            feat_dim=48,
            out_channels=2 # [Mask, Edge]
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
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        
        # 1. Shared 3D Backbone
        s1 = self.stage1(self.stem(x))      # (B, 24, 64, 64, 64)
        s2 = self.stage2(s1)                 # (B, 48, 32, 32, 32)
        s3 = self.stage3(s2)                 # (B, 96, 16, 16, 16)
        z = self.stage4(s3)                  # (B, 192, 8, 8, 8)
        skips = [s3, s2, s1]
        
        # 2. Forward 1: Initial Coarse Predictions & Edge Tokens
        coarse_liv = self.coarse_liver_head(z) # (B, 2, 8, 8, 8) [M0, E0]
        coarse_tum = self.coarse_tumor_head(z) # (B, 2, 8, 8, 8) [M0, E0]
        
        m0_liv = coarse_liv[:, 0:1]
        m0_tum = coarse_tum[:, 0:1]
        
        # Upsample coarse masks to 64^3 for RP-Encoder and prior loss
        m0_liv_64 = F.interpolate(m0_liv, size=(64, 64, 64), mode='trilinear', align_corners=False)
        m0_tum_64 = F.interpolate(m0_tum, size=(64, 64, 64), mode='trilinear', align_corners=False)
        
        # 3. Co-Seg RP-Encoder: Dense Region Prompts
        g_liver = self.rp_encoder_liver(torch.sigmoid(m0_liv_64), z) # (B, 192, 8, 8, 8)
        g_tumor = self.rp_encoder_tumor(torch.sigmoid(m0_tum_64), z) # (B, 192, 8, 8, 8)
        
        # 4. Co-Seg MP-Decoder: Mutual Query Guidance
        # Liver features conditioned by Tumor prompt
        z_liv = (z + g_tumor).flatten(2).permute(0, 2, 1) # (B, 512, 192)
        q_liv = self.query_liver.expand(B, -1, -1)
        q_liv_sa, _ = self.attn_liver_sa(q_liv, q_liv, q_liv)
        q_liv_ca, _ = self.attn_liver_ca(q_liv_sa, z_liv, z_liv)
        q_liv_ref = self.norm_liv_q(q_liv + q_liv_ca)
        h_liv, _ = self.attn_liver_rev(z_liv, q_liv_ref, q_liv_ref)
        h_liv_vol = self.norm_liv_feat(z_liv + h_liv).permute(0, 2, 1).view(B, self.embed_dim, 8, 8, 8)
        
        # Tumor features conditioned by Liver prompt
        z_tum = (z + g_liver).flatten(2).permute(0, 2, 1) # (B, 512, 192)
        q_tum = self.query_tumor.expand(B, -1, -1)
        q_tum_sa, _ = self.attn_tumor_sa(q_tum, q_tum, q_tum)
        q_tum_ca, _ = self.attn_tumor_ca(q_tum_sa, z_tum, z_tum)
        q_tum_ref = self.norm_tum_q(q_tum + q_tum_ca)
        h_tum, _ = self.attn_tumor_rev(z_tum, q_tum_ref, q_tum_ref)
        h_tum_vol = self.norm_tum_feat(z_tum + h_tum).permute(0, 2, 1).view(B, self.embed_dim, 8, 8, 8)
        
        # 5. SOPSeg PRM-3D: Multi-Scale Progressive Edge Refinement
        prm_liv_out = self.prm_liver(h_liv_vol, skips, x, coarse_liv)
        prm_tum_out = self.prm_tumor(h_tum_vol, skips, x, coarse_tum)
        
        # Full Resolution Predictions (64^3)
        liv_p64 = prm_liv_out['p64']
        tum_p64 = prm_tum_out['p64']
        
        refined_liver_mask = liv_p64[:, 0:1]
        refined_liver_edge = liv_p64[:, 1:2]
        refined_tumor_mask = tum_p64[:, 0:1]
        refined_tumor_edge = tum_p64[:, 1:2]
        
        subtypes = self.subtype_head(prm_tum_out['last_feat'])
        
        # Unified 6-Class Segmentation
        bg_logit = -refined_liver_mask
        liver_parenchyma = refined_liver_mask - refined_tumor_mask
        lesion_logits = subtypes + refined_tumor_mask
        full_seg = torch.cat([bg_logit, liver_parenchyma, lesion_logits], dim=1)
        
        # Multi-task Global Heads
        z_pool = self.pool(z).flatten(1)
        cls_logits = self.cls_head(z_pool)
        tumor_burden = self.burden_head(z_pool)
        
        return {
            'seg_logits': full_seg,                       # (B, 6, 64, 64, 64)
            'refined_liver_mask': refined_liver_mask,     # (B, 1, 64, 64, 64)
            'refined_liver_edge': refined_liver_edge,     # (B, 1, 64, 64, 64)
            'refined_tumor_mask': refined_tumor_mask,     # (B, 1, 64, 64, 64)
            'refined_tumor_edge': refined_tumor_edge,     # (B, 1, 64, 64, 64)
            'coarse_liver_logits': m0_liv_64,             # (B, 1, 64, 64, 64)
            'coarse_tumor_logits': m0_tum_64,             # (B, 1, 64, 64, 64)
            'p16_liver': prm_liv_out['p16'],
            'p32_liver': prm_liv_out['p32'],
            'p16_tumor': prm_tum_out['p16'],
            'p32_tumor': prm_tum_out['p32'],
            'cls_logits': cls_logits,                     # (B, 5)
            'tumor_burden': tumor_burden                  # (B, 1)
        }


if __name__ == '__main__':
    print("Testing SOPSegLiverNet...")
    sopseg = SOPSegLiverNet()
    x = torch.randn(2, 1, 64, 64, 64)
    out_sop = sopseg(x)
    print("SOPSegLiverNet seg_logits shape:", out_sop['seg_logits'].shape)
    
    print("\nTesting Combined CoSeg_SOPSeg_LiverNet...")
    coseg_sopseg = CoSeg_SOPSeg_LiverNet()
    out_comb = coseg_sopseg(x)
    print("CoSeg_SOPSeg seg_logits shape:", out_comb['seg_logits'].shape)
    print("CoSeg_SOPSeg refined_liver_mask shape:", out_comb['refined_liver_mask'].shape)
    print("CoSeg_SOPSeg refined_tumor_mask shape:", out_comb['refined_tumor_mask'].shape)
    print("CoSeg_SOPSeg refined_tumor_edge shape:", out_comb['refined_tumor_edge'].shape)
    print("CoSeg_SOPSeg cls_logits shape:", out_comb['cls_logits'].shape)
    
    params = sum(p.numel() for p in coseg_sopseg.parameters() if p.requires_grad)
    print(f"Total parameter count: {params / 1e6:.2f} M")
    print("All modules verified successfully!")
