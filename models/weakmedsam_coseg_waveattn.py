"""
Unified WeakMedSAM + Co-Seg + WaveAttn Multi-Task Architecture.
Combines:
1. 2.5D Multi-Slice Temporal/Spatial Continuity [z-1, z, z+1]
2. 2D Haar Discrete Wavelet Transform (HaarDWT2D) for high-frequency lesion edge preservation
3. Second-Order Bilinear Pooling (SOPSeg) for fine-grained tumor texture discrimination
4. Co-Seg Collaborative Mutual Guidance (MICCAI 2025) between Liver and Tumor streams
5. Sub-Class Exploration (SCE) multi-label intra-class representation from WeakMedSAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ---------------------------------------------------------------------------
# 1. 2D Haar Discrete Wavelet Transform (DWT2D)
# ---------------------------------------------------------------------------
class HaarDWT2D(nn.Module):
    """
    Decomposes (B, C, H, W) into:
    - LL: Low-frequency approximation (B, C, H/2, W/2)
    - LH, HL, HH: High-frequency horizontal, vertical, diagonal edges (B, 3*C, H/2, W/2)
    """
    def __init__(self):
        super().__init__()
        l = torch.tensor([1.0, 1.0]) / 1.41421356
        h = torch.tensor([-1.0, 1.0]) / 1.41421356
        ll = torch.outer(l, l)
        lh = torch.outer(l, h)
        hl = torch.outer(h, l)
        hh = torch.outer(h, h)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1) # [4, 1, 2, 2]
        self.register_buffer("filters", filters.float())

    def forward(self, x):
        b, c, h, w = x.shape
        w_filt = self.filters.repeat(c, 1, 1, 1)
        out = F.conv2d(x, w_filt, stride=2, groups=c) # [B, 4*C, H/2, W/2]
        h2, w2 = out.shape[-2:]
        out = out.view(b, c, 4, h2, w2)
        ll = out[:, :, 0]
        high = out[:, :, 1:].reshape(b, c * 3, h2, w2)
        return ll, high

# ---------------------------------------------------------------------------
# 2. Second-Order Pooling (SOPSeg) Texture Module
# ---------------------------------------------------------------------------
class SOPSegTextureBlock(nn.Module):
    """
    Captures second-order feature statistics (bilinear covariance) to distinguish
    heterogeneous tumor tissue textures from normal liver parenchyma.
    """
    def __init__(self, in_c, reduced_c=64):
        super().__init__()
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_c, reduced_c, kernel_size=1),
            nn.BatchNorm2d(reduced_c),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(reduced_c, in_c),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        feat = self.conv_reduce(x) # [B, reduced_c, H, W]
        r = feat.shape[1]
        flat = feat.view(b, r, -1) # [B, r, N]
        # Covariance pooling: [B, r, r]
        cov = torch.bmm(flat, flat.transpose(1, 2)) / (h * w)
        # Channel-wise diagonal energy
        diag = torch.diagonal(cov, dim1=1, dim2=2) # [B, r]
        # Attentive modulation vector
        weights = self.fc(diag).view(b, c, 1, 1)
        return x * (1.0 + weights)

# ---------------------------------------------------------------------------
# 3. Wavelet-Gated Attention Skip Connection
# ---------------------------------------------------------------------------
class WaveletAttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.conv_high = nn.Sequential(
            nn.Conv2d(f_l * 3, f_int, kernel_size=1),
            nn.BatchNorm2d(f_int),
            nn.ReLU(inplace=True)
        )
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # High-frequency edge bonus from DWT
        _, high = self.dwt(x)
        high_up = F.interpolate(self.conv_high(high), size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g1 + x1 + high_up)
        attn = self.psi(psi)
        return x * attn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# ---------------------------------------------------------------------------
# 4. Full Combined Architecture
# ---------------------------------------------------------------------------
class WeakMedSAM_CoSeg_WaveAttn_Net(nn.Module):
    """
    Combined State-of-the-Art Architecture:
    WeakMedSAM Sub-Class Exploration + Co-Seg Mutual Guidance + Wavelet Attention + SOPSeg.
    """
    def __init__(self, in_channels=3, child_classes=4, pretrained=True):
        super().__init__()
        self.child_classes = child_classes

        # Pretrained Multi-scale Encoder
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)

        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        ) # [B, 64, 128, 128]
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1 # [B, 64, 64, 64]
        self.layer2 = backbone.layer2 # [B, 128, 32, 32]
        self.layer3 = backbone.layer3 # [B, 256, 16, 16]
        self.layer4 = backbone.layer4 # [B, 512, 8, 8]

        # SOPSeg Texture Enhancement at Bottleneck
        self.sopseg = SOPSegTextureBlock(512, reduced_c=64)

        # WeakMedSAM Multi-Task Classification Heads
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.primary_cls_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2) # [Liver, Tumor]
        )
        self.subclass_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2 * child_classes + 1) # K=4 Liver + K=4 Tumor + bg
        )

        # Wavelet Attention Gates
        self.att4 = WaveletAttentionGate(f_g=256, f_l=256, f_int=128)
        self.att3 = WaveletAttentionGate(f_g=128, f_l=128, f_int=64)
        self.att2 = WaveletAttentionGate(f_g=64, f_l=64, f_int=32)
        self.att1 = WaveletAttentionGate(f_g=32, f_l=64, f_int=32)

        # Upsampling and Convolutions
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64 + 32, 32)

        # Co-Seg Mutual Guidance Modules (Liver <-> Tumor)
        self.liver_stream = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.tumor_stream = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        # Mutual Prompting: Liver informs Tumor, Tumor informs Liver
        self.liver_to_tumor_gate = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=1),
            nn.Sigmoid()
        )
        self.tumor_to_liver_gate = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=1),
            nn.Sigmoid()
        )

        # Final specialized segmentation heads
        self.head_liver = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.head_tumor = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        x0 = self.initial(x)       # [B, 64, 128, 128]
        p0 = self.maxpool(x0)      # [B, 64, 64, 64]
        x1 = self.layer1(p0)       # [B, 64, 64, 64]
        x2 = self.layer2(x1)       # [B, 128, 32, 32]
        x3 = self.layer3(x2)       # [B, 256, 16, 16]
        x4 = self.layer4(x3)       # [B, 512, 8, 8]

        # SOPSeg Texture Enhancement
        x4_sop = self.sopseg(x4)   # [B, 512, 8, 8]

        # Classification logits
        feat_pool = self.avgpool(x4_sop).flatten(1)
        p_cls = self.primary_cls_head(feat_pool) # [B, 2]
        s_cls = self.subclass_head(feat_pool)    # [B, 9]

        # Decoder with Wavelet Attention
        d4 = self.up4(x4_sop)
        a4 = self.att4(g=d4, x=x3)
        d4 = self.conv4(torch.cat([a4, d4], dim=1)) # [B, 256, 16, 16]

        d3 = self.up3(d4)
        a3 = self.att3(g=d3, x=x2)
        d3 = self.conv3(torch.cat([a3, d3], dim=1)) # [B, 128, 32, 32]

        d2 = self.up2(d3)
        a2 = self.att2(g=d2, x=x1)
        d2 = self.conv2(torch.cat([a2, d2], dim=1)) # [B, 64, 64, 64]

        d1 = self.up1(d2)
        a1 = self.att1(g=d1, x=x0)
        d1 = self.conv1(torch.cat([a1, d1], dim=1)) # [B, 32, 128, 128]

        # Upsample to 256x256
        f_shared = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False) # [B, 32, 256, 256]

        # Co-Seg Dual Collaborative Streams
        f_liv = self.liver_stream(f_shared)
        f_tum = self.tumor_stream(f_shared)

        # Mutual Guidance:
        # Tumor is conditioned on Liver presence (restricting tumor to liver parenchyma)
        f_tum_guided = f_tum * self.liver_to_tumor_gate(f_liv)
        # Liver is conditioned on Tumor (preventing liver mask from carving out tumor cavities)
        f_liv_guided = f_liv + 0.5 * (f_tum * self.tumor_to_liver_gate(f_tum))

        logit_liv = self.head_liver(f_liv_guided)
        logit_tum = self.head_tumor(f_tum_guided)

        seg_logits = torch.cat([logit_liv, logit_tum], dim=1) # [B, 2, 256, 256]

        return {
            "seg_logits": seg_logits,
            "primary_cls": p_cls,
            "subclass_cls": s_cls,
            "latent_feat": x4_sop
        }
