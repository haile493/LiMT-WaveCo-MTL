import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    """Dual 3D Convolution with InstanceNorm and LeakyReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ChannelAttention3D(nn.Module):
    """Squeeze-and-Excitation Channel Attention for 3D Feature Maps."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_ch = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_ch, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.fc(x)
        return x * weights

class InterNetworkFusionModule(nn.Module):
    """
    Inter-Network Channel Feature Fusion (IFFM) inspired by Chen et al. (ICONIP 2023).
    Bridges hierarchical encoder representations to both task heads using channel recalibration.
    """
    def __init__(self, bottleneck_ch):
        super().__init__()
        self.ca = ChannelAttention3D(bottleneck_ch)
        self.proj_seg = nn.Sequential(
            nn.Conv3d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(bottleneck_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.proj_cls = nn.Sequential(
            nn.Conv3d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(bottleneck_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, bottleneck_feat):
        # Apply channel attention
        attn_feat = self.ca(bottleneck_feat)
        seg_feat = self.proj_seg(attn_feat + bottleneck_feat)
        cls_feat = self.proj_cls(attn_feat + bottleneck_feat)
        return seg_feat, cls_feat

class MultiTaskLiverNet(nn.Module):
    """
    Multi-Task 3D Network for Joint Liver Segmentation & Lesion Classification.
    Shared Encoder -> Inter-network Feature Fusion -> (Seg Decoder, Cls Head).
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
        
        # Shared 3D Encoder
        self.enc1 = ConvBlock3D(in_channels, c)
        self.down1 = nn.MaxPool3d(2)
        
        self.enc2 = ConvBlock3D(c, c * 2)
        self.down2 = nn.MaxPool3d(2)
        
        self.enc3 = ConvBlock3D(c * 2, c * 4)
        self.down3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(c * 4, c * 8)
        
        # Inter-network Channel Feature Fusion
        self.fusion = InterNetworkFusionModule(c * 8)
        
        # Segmentation Decoder
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(c * 8, c * 4)
        
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(c * 4, c * 2)
        
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(c * 2, c)
        
        self.seg_head = nn.Conv3d(c, num_seg_classes, kernel_size=1)
        
        # Classification Head
        self.cls_pool_avg = nn.AdaptiveAvgPool3d(1)
        self.cls_pool_max = nn.AdaptiveMaxPool3d(1)
        
        self.cls_head = nn.Sequential(
            nn.Linear(c * 8 * 2, c * 4),
            nn.BatchNorm1d(c * 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(c * 4, num_cls_classes)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)         # c
        e2 = self.enc2(self.down1(e1)) # 2c
        e3 = self.enc3(self.down2(e2)) # 4c
        b = self.bottleneck(self.down3(e3)) # 8c
        
        # Inter-network Feature Fusion
        seg_b, cls_b = self.fusion(b)
        
        # Segmentation Branch
        d3 = self.up3(seg_b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        seg_logits = self.seg_head(d1)
        
        # Classification Branch
        avg_p = self.cls_pool_avg(cls_b).flatten(1)
        max_p = self.cls_pool_max(cls_b).flatten(1)
        cls_feat = torch.cat([avg_p, max_p], dim=1)
        cls_logits = self.cls_head(cls_feat)
        
        return {
            'seg_logits': seg_logits,
            'cls_logits': cls_logits
        }

class SingleTaskSegmentationNet(nn.Module):
    """Segmentation-Only 3D U-Net baseline."""
    def __init__(self, in_channels=1, num_seg_classes=6, base_channels=16):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock3D(in_channels, c)
        self.down1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(c, c * 2)
        self.down2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(c * 2, c * 4)
        self.down3 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(c * 4, c * 8)
        
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(c * 2, c)
        self.seg_head = nn.Conv3d(c, num_seg_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return {'seg_logits': self.seg_head(d1)}

class SingleTaskClassificationNet(nn.Module):
    """Classification-Only 3D CNN baseline."""
    def __init__(self, in_channels=1, num_cls_classes=5, base_channels=16):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock3D(in_channels, c)
        self.down1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(c, c * 2)
        self.down2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(c * 2, c * 4)
        self.down3 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(c * 4, c * 8)
        
        self.cls_pool_avg = nn.AdaptiveAvgPool3d(1)
        self.cls_pool_max = nn.AdaptiveMaxPool3d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(c * 8 * 2, c * 4),
            nn.BatchNorm1d(c * 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(c * 4, num_cls_classes)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        
        avg_p = self.cls_pool_avg(b).flatten(1)
        max_p = self.cls_pool_max(b).flatten(1)
        cls_feat = torch.cat([avg_p, max_p], dim=1)
        return {'cls_logits': self.cls_head(cls_feat)}

if __name__ == '__main__':
    x = torch.randn(2, 1, 64, 64, 64)
    model = MultiTaskLiverNet(base_channels=16)
    out = model(x)
    print("Multi-task seg logits:", out['seg_logits'].shape)
    print("Multi-task cls logits:", out['cls_logits'].shape)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
