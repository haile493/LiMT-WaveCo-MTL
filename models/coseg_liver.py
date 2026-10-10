"""
Co-Seg: Mutual Prompt-Guided Collaborative Learning for Volumetric Liver and Tumor Segmentation
Adapted from:
- Paper: "Co-Seg: Mutual Prompt-Guided Collaborative Learning for Tissue and Nuclei Segmentation" (MICCAI 2025)
- GitHub: https://github.com/xq141839/Co-Seg

Key Novelties adapted for 3D Liver CT Multi-Task:
1. Collaborative Co-Segmentation Paradigm:
   Liver (macro anatomy context) and Tumor (micro focal pathology) exhibit mutual conditional dependencies:
   p(y_liver, y_tumor | x) = p(y_liver | x) * p(y_tumor | y_liver, x) = p(y_tumor | x) * p(y_liver | y_tumor, x)
2. Region-Aware Prompt Encoder (RP-Encoder 3D):
   Encodes coarse mask logits through downsampling convolutions, 3D self-attention,
   and 3D cross-attention with shared image embeddings to generate dense structural prompts (g_liver, g_tumor).
3. Mutual Prompt Mask Decoder (MP-Decoder 3D):
   Enforces bidirectional cross-guidance:
   - Liver head is conditioned on the tumor prompt: z + g_tumor, refining organ boundaries and necrotic cavities.
   - Tumor head is conditioned on the liver prompt: z + g_liver, strictly pruning false positives outside the liver.
   Task-specific query embeddings (q_liver, q_tumor) undergo dual cross-attention to refine task features.
4. Calling Multi-Task Learning Models:
   Co-Seg can directly CALL any Multi-Task Learning (MTL) model (e.g. MultiTaskLiverNet, MultiTaskLiverNet_MSFF,
   ResVNetFusion, WeedSenseLiverNet). Co-Seg extracts the base MTL predictions (segmentation, classification, burden),
   processes them through RP-Encoder, and refines them through MP-Decoder for superior collaborative accuracy.
"""

import os
import sys
import math

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple, Dict, Union, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Residual 3D Convolutional Block with InstanceNorm3d and LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act2(out + res)
        return out


class RPEncoder3D(nn.Module):
    """
    3D Region-aware Prompt Encoder (RP-Encoder).
    Implements Equations (4) and (5) from the paper:
      m_i = Conv2(GELU(LN(Conv1(\mu_i))))
      g_i = CrossAttention(SelfAttention(m_i), z, z)
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Conv1: downsamples 64x64x64 mask logits to 8x8x8 matching image embedding z
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(embed_dim // 4, affine=True),
            nn.GELU(),
            nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(embed_dim // 2, affine=True),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(embed_dim, affine=True),
            nn.GELU(),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=True)
        )
        
        # Self-attention on mask prompt tokens
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_sa = nn.LayerNorm(embed_dim)
        
        # Cross-attention with shared image embeddings z
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_ca = nn.LayerNorm(embed_dim)
        
        # FFN refinement
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, mask_logits: torch.Tensor, z_embed: torch.Tensor) -> torch.Tensor:
        """
        mask_logits: (B, 1, D, H, W)
        z_embed: (B, C, D', H', W') where C == embed_dim
        returns:
          g_prompt: (B, C, D', H', W')
        """
        B, C, D_z, H_z, W_z = z_embed.shape
        N_vox = D_z * H_z * W_z
        
        # 1. Downsample and project mask
        m = self.downsample(mask_logits) # (B, C, D_z, H_z, W_z)
        
        # Flatten spatial dimensions to token sequences
        m_tokens = m.flatten(2).permute(0, 2, 1) # (B, N_vox, C)
        z_tokens = z_embed.flatten(2).permute(0, 2, 1) # (B, N_vox, C)
        
        # 2. Self-Attention
        sa_out, _ = self.self_attn(m_tokens, m_tokens, m_tokens)
        m_tokens = self.norm_sa(m_tokens + sa_out)
        
        # 3. Cross-Attention with shared image embedding z
        ca_out, _ = self.cross_attn(m_tokens, z_tokens, z_tokens)
        prompt_tokens = self.norm_ca(m_tokens + ca_out)
        
        # 4. FFN
        prompt_tokens = self.norm_ffn(prompt_tokens + self.ffn(prompt_tokens))
        
        # Reshape back to 3D volume
        g_prompt = prompt_tokens.permute(0, 2, 1).view(B, C, D_z, H_z, W_z)
        return g_prompt


class MPDecoderHead3D(nn.Module):
    """
    3D Mutual Prompt Mask Decoder Head.
    Implements Equations (6) and (7) from the paper:
      q'_i = CrossAttention(SelfAttention(q_i), z \oplus g_j, z \oplus g_j)
      h_i = CrossAttention(z \oplus g_j, q'_i, q'_i)
    Followed by progressive 3D volumetric upsampling with multi-scale skip connections.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_queries: int = 16,
        num_heads: int = 4,
        out_channels: int = 1,
        skip_channels: List[int] = [96, 48, 24] # S3, S2, S1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        
        # Learnable task query embeddings
        self.task_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        
        # Query Self-Attention
        self.query_sa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_q_sa = nn.LayerNorm(embed_dim)
        
        # Query Cross-Attention with mutual prompt conditioned features (z + g_j)
        self.query_ca = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_q_ca = nn.LayerNorm(embed_dim)
        
        self.query_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm_q_mlp = nn.LayerNorm(embed_dim)
        
        # Reverse Cross-Attention: Feature conditioned by refined queries
        self.feat_ca = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_feat_ca = nn.LayerNorm(embed_dim)
        
        # Volumetric Pixel Decoder with 3D upsampling and skip fusion
        # Step 1: 8x8x8 -> 16x16x16 (fuse S3)
        self.up1 = nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.fuse1 = ConvBlock3D(embed_dim // 2 + skip_channels[0], embed_dim // 2)
        
        # Step 2: 16x16x16 -> 32x32x32 (fuse S2)
        self.up2 = nn.ConvTranspose3d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2)
        self.fuse2 = ConvBlock3D(embed_dim // 4 + skip_channels[1], embed_dim // 4)
        
        # Step 3: 32x32x32 -> 64x64x64 (fuse S1)
        self.up3 = nn.ConvTranspose3d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2)
        self.fuse3 = ConvBlock3D(embed_dim // 8 + skip_channels[2], embed_dim // 8)
        
        # Final prediction head
        self.head = nn.Conv3d(embed_dim // 8, out_channels, kernel_size=1)

    def forward(
        self,
        z_embed: torch.Tensor,
        g_prompt_other: torch.Tensor,
        skips: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        z_embed: (B, C, 8, 8, 8)
        g_prompt_other: (B, C, 8, 8, 8)
        skips: [S3 (16^3), S2 (32^3), S1 (64^3)]
        """
        B, C, D_z, H_z, W_z = z_embed.shape
        N_vox = D_z * H_z * W_z
        
        # Modulate visual embedding with mutual prompt
        z_conditioned = z_embed + g_prompt_other
        feat_tokens = z_conditioned.flatten(2).permute(0, 2, 1) # (B, N_vox, C)
        
        # Expand task queries for batch
        queries = self.task_queries.expand(B, -1, -1) # (B, N_q, C)
        
        # 1. Query Self-Attention
        q_sa_out, _ = self.query_sa(queries, queries, queries)
        queries = self.norm_q_sa(queries + q_sa_out)
        
        # 2. Query Cross-Attention with conditioned visual tokens
        q_ca_out, _ = self.query_ca(queries, feat_tokens, feat_tokens)
        q_refined = self.norm_q_ca(queries + q_ca_out)
        q_refined = self.norm_q_mlp(q_refined + self.query_mlp(q_refined))
        
        # 3. Reverse Cross-Attention: refine visual tokens using queries
        h_tokens, _ = self.feat_ca(feat_tokens, q_refined, q_refined)
        h_tokens = self.norm_feat_ca(feat_tokens + h_tokens)
        
        # Reshape to 3D feature volume (B, C, 8, 8, 8)
        h_vol = h_tokens.permute(0, 2, 1).view(B, C, D_z, H_z, W_z)
        
        # 4. Volumetric Decoder with skip connections
        s3, s2, s1 = skips
        
        d1 = self.up1(h_vol)
        d1 = self.fuse1(torch.cat([d1, s3], dim=1))
        
        d2 = self.up2(d1)
        d2 = self.fuse2(torch.cat([d2, s2], dim=1))
        
        d3 = self.up3(d2)
        d3 = self.fuse3(torch.cat([d3, s1], dim=1))
        
        logits = self.head(d3)
        return logits


class CoSegLiverNet(nn.Module):
    """
    Co-Seg adapted for Liver and Tumor Multi-Task CT Segmentation.
    
    Now capable of CALLING an external Multi-Task Learning (MTL) model:
    - If `mtl_model` is provided (e.g. MultiTaskLiverNet, MultiTaskLiverNet_MSFF, WeedSenseLiverNet),
      Co-Seg executes the MTL model in Forward 1 to obtain initial coarse liver and tumor predictions,
      then applies RP-Encoder and MP-Decoder to mutually refine them in Forward 2.
    - If `mtl_model` is None, Co-Seg uses its internal 3D coarse multi-task heads.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 6,
        num_cls_classes: int = 5,
        base_channels: int = 24,
        mtl_model: Optional[Union[nn.Module, str]] = None
    ):
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes
        
        c1 = base_channels       # 24
        c2 = base_channels * 2   # 48
        c3 = base_channels * 4   # 96
        c4 = base_channels * 8   # 192
        self.embed_dim = c4
        
        # Shared 3D Backbone Encoder
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c1, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.stage1 = ConvBlock3D(c1, c1, stride=1) # 64x64x64
        self.stage2 = ConvBlock3D(c1, c2, stride=2) # 32x32x32
        self.stage3 = ConvBlock3D(c2, c3, stride=2) # 16x16x16
        self.stage4 = ConvBlock3D(c3, c4, stride=2) # 8x8x8
        
        # Forward 1 Built-in Coarse Decoders (used when no external MTL model is called)
        self.coarse_up = nn.Sequential(
            nn.ConvTranspose3d(c4, c2, kernel_size=4, stride=4),
            nn.InstanceNorm3d(c2, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2),
            nn.InstanceNorm3d(c1, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.coarse_liver_head = nn.Conv3d(c1, 1, kernel_size=1)
        self.coarse_tumor_head = nn.Conv3d(c1, 1, kernel_size=1)
        
        # Region-aware Prompt Encoders (RP-Encoder)
        self.rp_encoder_liver = RPEncoder3D(in_channels=1, embed_dim=c4, num_heads=4)
        self.rp_encoder_tumor = RPEncoder3D(in_channels=1, embed_dim=c4, num_heads=4)
        
        # Mutual Prompt Mask Decoder Heads (MP-Decoder)
        # Liver Head (binary liver vs background)
        self.mp_decoder_liver = MPDecoderHead3D(
            embed_dim=c4,
            num_queries=16,
            num_heads=4,
            out_channels=1,
            skip_channels=[c3, c2, c1]
        )
        
        # Tumor Head: outputs binary tumor (channel 0) and 4 lesion subtypes (channels 1..4: cyst, hemangioma, met, hcc)
        self.mp_decoder_tumor = MPDecoderHead3D(
            embed_dim=c4,
            num_queries=16,
            num_heads=4,
            out_channels=5, # 1 binary tumor + 4 subtype logits
            skip_channels=[c3, c2, c1]
        )
        
        # Multi-task Global Heads (Lesion Classification & Tumor Burden Regression)
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
        
        # Multi-Task Learning model integration
        self.mtl_model = None
        if mtl_model is not None:
            self.set_mtl_model(mtl_model)

    def set_mtl_model(self, mtl_model: Union[nn.Module, str], freeze_mtl: bool = True):
        """
        Attaches a multi-task learning model to be called in Forward 1.
        Can be an instantiated nn.Module or a path to a checkpoint.
        """
        if isinstance(mtl_model, str):
            # Checkpoint file path
            if os.path.exists(mtl_model):
                ckpt = torch.load(mtl_model, map_location='cpu', weights_only=False)
                state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                
                # Auto-detect architecture from checkpoint keys
                if any('dfe' in k or 'ecadd' in k for k in state.keys()):
                    from models.multitask_msff_net import MultiTaskLiverNet_MSFF
                    model_inst = MultiTaskLiverNet_MSFF(in_channels=1, num_seg_classes=6, num_cls_classes=5)
                elif any('uib' in k or 'bga' in k for k in state.keys()):
                    from models.weedsense_liver import WeedSenseLiverNet
                    model_inst = WeedSenseLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5)
                elif any('fusion' in k for k in state.keys()) and any('enc' in k for k in state.keys()):
                    from models.multitask_net import MultiTaskLiverNet
                    model_inst = MultiTaskLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5)
                else:
                    from models.multitask_net import MultiTaskLiverNet
                    model_inst = MultiTaskLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5)
                    
                model_inst.load_state_dict(state, strict=False)
                self.mtl_model = model_inst
            else:
                raise FileNotFoundError(f"Checkpoint not found: {mtl_model}")
        elif isinstance(mtl_model, nn.Module):
            self.mtl_model = mtl_model
        else:
            raise TypeError(f"Expected nn.Module or checkpoint path, got {type(mtl_model)}")
            
        if freeze_mtl:
            for p in self.mtl_model.parameters():
                p.requires_grad = False
            self.mtl_model.eval()

    def forward(
        self,
        x: torch.Tensor,
        external_coarse_liver: Optional[torch.Tensor] = None,
        external_coarse_tumor: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass executing collaborative multi-task learning.
        Can optionally call an external multi-task learning model.
        """
        # --- Shared 3D Backbone ---
        s1 = self.stage1(self.stem(x))      # (B, 24, 64, 64, 64)
        s2 = self.stage2(s1)                 # (B, 48, 32, 32, 32)
        s3 = self.stage3(s2)                 # (B, 96, 16, 16, 16)
        z = self.stage4(s3)                  # (B, 192, 8, 8, 8)
        
        # --- Forward 1: Coarse Multi-Task Mask Generation ---
        mtl_out = None
        if self.mtl_model is not None:
            # CALL the multi-task learning model directly
            mtl_out = self.mtl_model(x)
            
            if 'seg_logits' in mtl_out:
                seg_l = mtl_out['seg_logits']
                if seg_l.shape[1] == 2:
                    coarse_liver_logits = seg_l[:, 0:1]
                    coarse_tumor_logits = seg_l[:, 1:2]
                elif seg_l.shape[1] == 6:
                    p_all = F.softmax(seg_l, dim=1)
                    p_liv = p_all[:, 1:].sum(dim=1, keepdim=True).clamp(1e-4, 1.0 - 1e-4)
                    p_tum = p_all[:, 2:].sum(dim=1, keepdim=True).clamp(1e-4, 1.0 - 1e-4)
                    coarse_liver_logits = torch.logit(p_liv)
                    coarse_tumor_logits = torch.logit(p_tum)
                else:
                    coarse_liver_logits = seg_l[:, 0:1]
                    coarse_tumor_logits = seg_l[:, 1:2]
            else:
                feat_coarse = self.coarse_up(z)
                coarse_liver_logits = self.coarse_liver_head(feat_coarse)
                coarse_tumor_logits = self.coarse_tumor_head(feat_coarse)
        elif external_coarse_liver is not None and external_coarse_tumor is not None:
            coarse_liver_logits = external_coarse_liver
            coarse_tumor_logits = external_coarse_tumor
        else:
            feat_coarse = self.coarse_up(z) # (B, 24, 64, 64, 64)
            coarse_liver_logits = self.coarse_liver_head(feat_coarse)
            coarse_tumor_logits = self.coarse_tumor_head(feat_coarse)
            
        # --- RP-Encoder: Dense Region Prompts ---
        # Liver prompt from coarse liver logits
        g_liver = self.rp_encoder_liver(torch.sigmoid(coarse_liver_logits), z)
        # Tumor prompt from coarse tumor logits
        g_tumor = self.rp_encoder_tumor(torch.sigmoid(coarse_tumor_logits), z)
        
        # --- Forward 2: Mutual Prompt Guidance (MP-Decoder) ---
        skips = [s3, s2, s1]
        # Liver Head guided by Tumor prompt (z + g_tumor)
        refined_liver_logits = self.mp_decoder_liver(z, g_tumor, skips) # (B, 1, 64, 64, 64)
        
        # Tumor Head guided by Liver prompt (z + g_liver)
        refined_tumor_outputs = self.mp_decoder_tumor(z, g_liver, skips) # (B, 5, 64, 64, 64)
        refined_tumor_binary = refined_tumor_outputs[:, 0:1]              # (B, 1, 64, 64, 64)
        refined_tumor_subtypes = refined_tumor_outputs[:, 1:5]            # (B, 4, 64, 64, 64)
        
        # --- Multi-class 6-channel unification for LiMT benchmark ---
        # Channels: [0: background, 1: liver, 2: cyst, 3: hemangioma, 4: met, 5: hcc]
        p_liver = torch.sigmoid(refined_liver_logits)
        p_tumor = torch.sigmoid(refined_tumor_binary)
        
        bg_logit = -refined_liver_logits
        liver_parenchyma_logit = refined_liver_logits - refined_tumor_binary
        lesion_logits = refined_tumor_subtypes + refined_tumor_binary
        
        full_seg_logits = torch.cat([bg_logit, liver_parenchyma_logit, lesion_logits], dim=1) # (B, 6, 64, 64, 64)
        
        # --- Global Multi-Task Heads ---
        z_pool = self.pool(z).flatten(1)
        cls_logits = self.cls_head(z_pool)
        tumor_burden = self.burden_head(z_pool)
        
        # If external MTL model was called, fuse or supply its classification & burden
        if mtl_out is not None:
            if 'cls_logits' in mtl_out:
                cls_logits = 0.5 * cls_logits + 0.5 * mtl_out['cls_logits']
            if 'tumor_burden' in mtl_out:
                tumor_burden = 0.5 * tumor_burden + 0.5 * mtl_out['tumor_burden']
        
        return {
            'seg_logits': full_seg_logits,                # 6-class unified logits for evaluation
            'refined_liver_logits': refined_liver_logits, # (B, 1, 64, 64, 64)
            'refined_tumor_binary': refined_tumor_binary, # (B, 1, 64, 64, 64)
            'refined_tumor_subtypes': refined_tumor_subtypes, # (B, 4, 64, 64, 64)
            'coarse_liver_logits': coarse_liver_logits,   # (B, 1, 64, 64, 64)
            'coarse_tumor_logits': coarse_tumor_logits,   # (B, 1, 64, 64, 64)
            'cls_logits': cls_logits,                     # (B, 5)
            'tumor_burden': tumor_burden,                 # (B, 1)
            'mtl_outputs': mtl_out                        # Raw outputs from called MTL model
        }


class CoSegMultiTaskWrapper(nn.Module):
    """
    Universal Collaborative Wrapper allowing Co-Seg to wrap and call ANY multi-task learning model.
    Usage:
        mtl_net = MultiTaskLiverNet()
        coseg = CoSegMultiTaskWrapper(mtl_net)
        out = coseg(images)
    """
    def __init__(self, mtl_model: nn.Module, base_channels: int = 24):
        super().__init__()
        self.model = CoSegLiverNet(
            in_channels=1,
            num_seg_classes=6,
            num_cls_classes=5,
            base_channels=base_channels,
            mtl_model=mtl_model
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, base_channels: int = 24):
        inst = cls(mtl_model=None, base_channels=base_channels)
        inst.model.set_mtl_model(ckpt_path)
        return inst


if __name__ == '__main__':
    from models.multitask_net import MultiTaskLiverNet
    
    print("Testing Co-Seg with internal multi-task heads...")
    model_standalone = CoSegLiverNet()
    x = torch.randn(2, 1, 64, 64, 64)
    out1 = model_standalone(x)
    print("Standalone Co-Seg seg_logits shape:", out1['seg_logits'].shape)
    
    print("\nTesting Co-Seg CALLING external MultiTaskLiverNet model...")
    mtl_baseline = MultiTaskLiverNet()
    model_with_mtl = CoSegLiverNet(mtl_model=mtl_baseline)
    out2 = model_with_mtl(x)
    print("Co-Seg calling MultiTaskLiverNet seg_logits shape:", out2['seg_logits'].shape)
    print("Co-Seg calling MultiTaskLiverNet coarse_liver shape:", out2['coarse_liver_logits'].shape)
    print("Co-Seg calling MultiTaskLiverNet cls_logits shape:", out2['cls_logits'].shape)
    print("Has mtl_outputs:", out2['mtl_outputs'] is not None)
    print("\nAll Multi-Task Calling tests passed successfully!")
