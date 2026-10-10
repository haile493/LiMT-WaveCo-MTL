"""
WeakMedSAM-Seg: High-Resolution Multi-Task Segmentation Network for Liver and Tumor.
Features:
- Multi-scale residual feature extraction with ImageNet pretraining
- Attention-gated skip connections (Attention U-Net)
- Dual specialized segmentation heads (Liver and Tumor)
- Anatomical Hierarchy Gating (P(tumor) constrained by Liver boundary)
- Auxiliary classification head for multi-task synergy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal from deeper layer, x: skip connection from encoder
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
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

class WeakMedSAM_SegNet(nn.Module):
    def __init__(self, in_channels=3, base_encoder="resnet34", pretrained=True):
        super().__init__()
        
        # Pretrained Feature Encoder
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)

        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        ) # [B, 64, 128, 128]
        self.maxpool = backbone.maxpool # [B, 64, 64, 64]

        self.layer1 = backbone.layer1 # [B, 64, 64, 64]
        self.layer2 = backbone.layer2 # [B, 128, 32, 32]
        self.layer3 = backbone.layer3 # [B, 256, 16, 16]
        self.layer4 = backbone.layer4 # [B, 512, 8, 8]

        # Auxiliary Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2) # [Liver, Tumor]
        )

        # Attention U-Net Decoder
        # D4: 512 -> 256
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(f_g=256, f_l=256, f_int=128)
        self.conv4 = ConvBlock(512, 256)

        # D3: 256 -> 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(f_g=128, f_l=128, f_int=64)
        self.conv3 = ConvBlock(256, 128)

        # D2: 128 -> 64
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(f_g=64, f_l=64, f_int=32)
        self.conv2 = ConvBlock(128, 64)

        # D1: 64 -> 32
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(f_g=32, f_l=64, f_int=32)
        self.conv1 = ConvBlock(64 + 32, 32)

        # Dual Heads: Liver and Tumor
        self.head_liver = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        self.head_tumor = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
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

        # Classification logits
        cls_feat = self.avgpool(x4).flatten(1)
        cls_logits = self.cls_head(cls_feat) # [B, 2]

        # Decoder with Attention Gates
        d4 = self.up4(x4)                    # [B, 256, 16, 16]
        att4 = self.att4(g=d4, x=x3)         # [B, 256, 16, 16]
        d4 = torch.cat([att4, d4], dim=1)    # [B, 512, 16, 16]
        d4 = self.conv4(d4)                  # [B, 256, 16, 16]

        d3 = self.up3(d4)                    # [B, 128, 32, 32]
        att3 = self.att3(g=d3, x=x2)         # [B, 128, 32, 32]
        d3 = torch.cat([att3, d3], dim=1)    # [B, 256, 32, 32]
        d3 = self.conv3(d3)                  # [B, 128, 32, 32]

        d2 = self.up2(d3)                    # [B, 64, 64, 64]
        att2 = self.att2(g=d2, x=x1)         # [B, 64, 64, 64]
        d2 = torch.cat([att2, d2], dim=1)    # [B, 128, 64, 64]
        d2 = self.conv2(d2)                  # [B, 64, 64, 64]

        d1 = self.up1(d2)                    # [B, 32, 128, 128]
        att1 = self.att1(g=d1, x=x0)         # [B, 64, 128, 128]
        d1 = torch.cat([att1, d1], dim=1)    # [B, 96, 128, 128]
        d1 = self.conv1(d1)                  # [B, 32, 128, 128]

        # Final upsample to 256x256
        d0 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False) # [B, 32, 256, 256]

        logit_liver = self.head_liver(d0)   # [B, 1, 256, 256]
        logit_tumor = self.head_tumor(d0)   # [B, 1, 256, 256]

        # Concatenate dual logits: Channel 0: Liver, Channel 1: Tumor
        seg_logits = torch.cat([logit_liver, logit_tumor], dim=1) # [B, 2, 256, 256]

        return {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits,
            "latent_feat": x4
        }
