import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECModule(nn.Module):
    """
    Efficient Channel (EC) module from EDNet (Hu et al., Scientific Reports 2025).
    Computes adaptive channel weights without dimensionality reduction:
      k = |(log2(C) + b) / r| (Eq. 7 in paper, r=2, b=1)
      w' = sigmoid(Conv1d(AAP(X)))
      X_EC = X * w'
    """
    def __init__(self, channels, r=2, b=1):
        super().__init__()
        # Determine kernel size k (must be odd integer)
        t = int(abs((math.log2(max(channels, 2)) + b) / r))
        k = t if t % 2 == 1 else t + 1
        k = max(k, 3)
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 1D conv over channel dimension
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, D, H, W)
        b, c, _, _, _ = x.shape
        # Pool to (B, C, 1, 1, 1) -> (B, 1, C)
        z = self.avg_pool(x).view(b, 1, c)
        w = self.conv1d(z) # (B, 1, C)
        w_prime = self.sigmoid(w).view(b, c, 1, 1, 1)
        return x * w_prime

class ECAddBlock(nn.Module):
    """
    Residual block with Automated Feature Fusion Module (ECAdd) from EDNet.
    Fuses shallow local features (conv1) and deep semantic features (conv2)
    via dual EC modules and weighted element-wise addition:
      X_ECAdd = EC1(X_shallow) + EC2(X_deep)
      Output = LeakyReLU(X_ECAdd + Shortcut(X))
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.ec1 = ECModule(out_channels)
        self.ec2 = ECModule(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        # Adaptive feature fusion
        x1_ec = self.ec1(x1)
        x2_ec = self.ec2(x2)
        x_fused = x1_ec + x2_ec
        
        return self.act(x_fused + res)

class DFEAttention(nn.Module):
    """
    Deep Feature Enhancement (DFE) Attention Module from EDNet.
    Fuses multi-scale features (shallower X1 and deeper X2) using:
      1. Trilinear interpolation of X2 to size of X1
      2. Concatenation along channel dimension
      3. 1x1x1 conv channel fusion: F = ReLU(Conv1x1x1(Concat(X1, X2')))
      4. Spatial-channel attention: Attn = sigmoid(Conv3x3x3(ReLU(Conv3x3x3(F))))
      5. Output = F * Attn
    """
    def __init__(self, in_channels_shallow, in_channels_deep, out_channels):
        super().__init__()
        total_ch = in_channels_shallow + in_channels_deep
        self.conv_fuse = nn.Sequential(
            nn.Conv3d(total_ch, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
        self.attn_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_shallow, x_deep):
        # Upsample deep feature to match shallow feature spatial dimensions
        if x_deep.shape[2:] != x_shallow.shape[2:]:
            x_deep_up = F.interpolate(x_deep, size=x_shallow.shape[2:], mode='trilinear', align_corners=False)
        else:
            x_deep_up = x_deep
            
        f_cat = torch.cat([x_shallow, x_deep_up], dim=1)
        f = self.conv_fuse(f_cat)
        attn = self.attn_conv(f)
        return f * attn

class MultiTaskLiverNet_MSFF(nn.Module):
    """
    Enhanced Multi-Task Liver & Tumor Network integrating:
    1. Residual Encoder with ECAdd (Automated Multi-Scale Feature Fusion)
    2. Convolutional Downsampling to prevent small-lesion information loss
    3. Deep Feature Enhancement (DFE) Attention at Decoder Skip Connections
    4. Inter-network Channel Feature Fusion for Multi-Label Classification
    5. Multi-Scale Deep Supervision Auxiliary Segmentation Head
    """
    def __init__(
        self,
        in_channels=1,
        num_seg_classes=6,
        num_cls_classes=5,
        base_channels=16
    ):
        super().__init__()
        c = base_channels
        
        # Encoder Stage 1
        self.enc1 = ECAddBlock(in_channels, c)
        # Conv downsampling instead of maxpool to preserve semantic and boundary details
        self.down1 = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # Encoder Stage 2
        self.enc2 = ECAddBlock(c, c * 2)
        self.down2 = nn.Sequential(
            nn.Conv3d(c * 2, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(c * 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # Encoder Stage 3
        self.enc3 = ECAddBlock(c * 2, c * 4)
        self.down3 = nn.Sequential(
            nn.Conv3d(c * 4, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(c * 4, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = ECAddBlock(c * 4, c * 8)
        
        # DFE Attention at decoder skip connections
        self.dfe3 = DFEAttention(in_channels_shallow=c * 4, in_channels_deep=c * 8, out_channels=c * 4)
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ECAddBlock(c * 4 + c * 4, c * 4)
        
        self.dfe2 = DFEAttention(in_channels_shallow=c * 2, in_channels_deep=c * 4, out_channels=c * 2)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ECAddBlock(c * 2 + c * 2, c * 2)
        
        self.dfe1 = DFEAttention(in_channels_shallow=c, in_channels_deep=c * 2, out_channels=c)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ECAddBlock(c + c, c)
        
        # Final Segmentation Head
        self.seg_head = nn.Conv3d(c, num_seg_classes, kernel_size=1)
        
        # Deep Supervision Auxiliary Head at Stage 2
        self.aux_seg_head = nn.Sequential(
            nn.Conv3d(c * 2, num_seg_classes, kernel_size=1)
        )
        
        # Inter-network Channel Feature Fusion for Multi-Label Classification Head
        self.cls_pool_avg = nn.AdaptiveAvgPool3d(1)
        self.cls_pool_max = nn.AdaptiveMaxPool3d(1)
        
        # Combines bottleneck features (c*8) and decoder deep features (c*4)
        self.cls_head = nn.Sequential(
            nn.Linear((c * 8 + c * 4) * 2, c * 4),
            nn.BatchNorm1d(c * 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(c * 4, num_cls_classes)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # (B, c, D, H, W)
        e2 = self.enc2(self.down1(e1)) # (B, 2c, D/2, H/2, W/2)
        e3 = self.enc3(self.down2(e2)) # (B, 4c, D/4, H/4, W/4)
        b = self.bottleneck(self.down3(e3)) # (B, 8c, D/8, H/8, W/8)
        
        # Decoder Stage 3 with DFE attention
        e3_enhanced = self.dfe3(e3, b)
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3_enhanced], dim=1))
        
        # Decoder Stage 2 with DFE attention & auxiliary supervision
        e2_enhanced = self.dfe2(e2, d3)
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2_enhanced], dim=1))
        
        # Deep supervision output (upsampled to match input spatial size)
        aux_logits = F.interpolate(self.aux_seg_head(d2), size=x.shape[2:], mode='trilinear', align_corners=False)
        
        # Decoder Stage 1 with DFE attention
        e1_enhanced = self.dfe1(e1, d2)
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1_enhanced], dim=1))
        
        seg_logits = self.seg_head(d1)
        
        # Classification Branch
        b_avg = self.cls_pool_avg(b).flatten(1)
        b_max = self.cls_pool_max(b).flatten(1)
        d3_avg = self.cls_pool_avg(d3).flatten(1)
        d3_max = self.cls_pool_max(d3).flatten(1)
        
        cls_feat = torch.cat([b_avg, b_max, d3_avg, d3_max], dim=1)
        cls_logits = self.cls_head(cls_feat)
        
        return {
            'seg_logits': seg_logits,
            'aux_seg_logits': aux_logits,
            'cls_logits': cls_logits
        }

if __name__ == '__main__':
    x = torch.randn(2, 1, 64, 64, 64)
    model = MultiTaskLiverNet_MSFF(base_channels=16)
    out = model(x)
    print("Seg logits:", out['seg_logits'].shape)
    print("Aux seg logits:", out['aux_seg_logits'].shape)
    print("Cls logits:", out['cls_logits'].shape)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
