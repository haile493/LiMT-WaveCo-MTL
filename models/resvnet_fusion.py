import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel Attention Module from Ref [78].
    Automatically evaluates the importance of channel features in the segmentation network.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_ch = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_ch, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class InterNetworkChannelFeatureFusion(nn.Module):
    """
    Inter-Network Channel Feature Fusion (ICFF) from Ref [78] (Section 3.3, Fig. 3).
    1. Channel importance evaluated via SE structure on segmentation features: F_seg' = SE(F_seg)
    2. Spatial dimension aligned (downsampled/upsampled) to match classification features
    3. Concatenated along channel dimension: Concat([F_seg', F_cls])
    4. Conv3D layer reduces channel dimension back to classification stage channels.
    """
    def __init__(self, seg_channels, cls_channels, out_channels):
        super().__init__()
        self.se = SEBlock3D(seg_channels)
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(seg_channels + cls_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_seg, f_cls):
        f_seg_se = self.se(f_seg)
        if f_seg_se.shape[2:] != f_cls.shape[2:]:
            f_seg_aligned = F.interpolate(f_seg_se, size=f_cls.shape[2:], mode='trilinear', align_corners=False)
        else:
            f_seg_aligned = f_seg_se
        f_cat = torch.cat([f_seg_aligned, f_cls], dim=1)
        return self.fuse_conv(f_cat)

class VNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            ch_in = in_channels if i == 0 else out_channels
            layers.extend([
                nn.Conv3d(ch_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            ])
        self.conv = nn.Sequential(*layers)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class ResNetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.in1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm3d(out_channels, affine=True)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        return self.relu(out + res)

class ResVNetFusion(nn.Module):
    """
    Complete Multi-Task Learning Network from Ref [78]:
    "Multi-task Learning Network for Automatic Pancreatic Tumor Segmentation and Classification
     with Inter-Network Channel Feature Fusion" (Chen et al., ICONIP 2023)
    
    1. V-Net Segmentation Sub-network: outputs [Liver, Tumor] masks (or Tumor only)
    2. Soft Tumor Region Masking: X_masked = X * Sigmoid(P_tumor)
    3. Inter-Network Channel Feature Fusion (ICFF) with SE channel recalibration
    4. ResNet Classification Sub-network taking X_masked and fused features
    """
    def __init__(
        self,
        in_channels=1,
        num_seg_classes=2, # Channel 0: Liver, Channel 1: Tumor
        num_cls_classes=5, # Multi-label disease classification
        base_channels=16
    ):
        super().__init__()
        c = base_channels
        self.num_seg_classes = num_seg_classes
        
        # ==========================================
        # 1. V-Net Segmentation Sub-network
        # ==========================================
        self.seg_enc1 = VNetEncoderBlock(in_channels, c, n_convs=1)
        self.seg_down1 = nn.Conv3d(c, c * 2, kernel_size=2, stride=2)
        
        self.seg_enc2 = VNetEncoderBlock(c * 2, c * 2, n_convs=2)
        self.seg_down2 = nn.Conv3d(c * 2, c * 4, kernel_size=2, stride=2)
        
        self.seg_enc3 = VNetEncoderBlock(c * 4, c * 4, n_convs=2)
        self.seg_down3 = nn.Conv3d(c * 4, c * 8, kernel_size=2, stride=2)
        
        self.seg_bottleneck = VNetEncoderBlock(c * 8, c * 8, n_convs=3)
        
        self.seg_up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.seg_dec3 = VNetEncoderBlock(c * 8, c * 4, n_convs=2)
        
        self.seg_up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.seg_dec2 = VNetEncoderBlock(c * 4, c * 2, n_convs=2)
        
        self.seg_up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.seg_dec1 = VNetEncoderBlock(c * 2, c, n_convs=1)
        
        self.seg_out = nn.Conv3d(c, num_seg_classes, kernel_size=1)
        
        # ==========================================
        # 2. Inter-Network Channel Feature Fusion (ICFF)
        # ==========================================
        self.icff1 = InterNetworkChannelFeatureFusion(seg_channels=c * 2, cls_channels=c * 2, out_channels=c * 2)
        self.icff2 = InterNetworkChannelFeatureFusion(seg_channels=c * 4, cls_channels=c * 4, out_channels=c * 4)
        self.icff3 = InterNetworkChannelFeatureFusion(seg_channels=c * 8, cls_channels=c * 8, out_channels=c * 8)
        
        # ==========================================
        # 3. 3D ResNet Classification Sub-network
        # ==========================================
        self.cls_init = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.cls_stage1 = ResNetBlock3D(c, c * 2, stride=2)
        self.cls_stage2 = ResNetBlock3D(c * 2, c * 4, stride=2)
        self.cls_stage3 = ResNetBlock3D(c * 4, c * 8, stride=2)
        
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(c * 8, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(c * 4, num_cls_classes)
        )

    def forward(self, x):
        # ------------------------------------------
        # 1. Forward V-Net Segmentation Network
        # ------------------------------------------
        s1 = self.seg_enc1(x)              # (B, c, D, H, W)
        s2 = self.seg_enc2(self.seg_down1(s1)) # (B, 2c, D/2, H/2, W/2)
        s3 = self.seg_enc3(self.seg_down2(s2)) # (B, 4c, D/4, H/4, W/4)
        sb = self.seg_bottleneck(self.seg_down3(s3)) # (B, 8c, D/8, H/8, W/8)
        
        d3 = self.seg_up3(sb)
        d3 = self.seg_dec3(torch.cat([d3, s3], dim=1)) # (B, 4c, D/4, H/4, W/4)
        
        d2 = self.seg_up2(d3)
        d2 = self.seg_dec2(torch.cat([d2, s2], dim=1)) # (B, 2c, D/2, H/2, W/2)
        
        d1 = self.seg_up1(d2)
        d1 = self.seg_dec1(torch.cat([d1, s1], dim=1)) # (B, c, D, H, W)
        
        seg_logits = self.seg_out(d1) # (B, num_seg_classes, D, H, W)
        
        # ------------------------------------------
        # 2. Tumor Masking: X_masked = X * Sigmoid(P_tumor)
        # ------------------------------------------
        if self.num_seg_classes >= 2:
            p_tumor = torch.sigmoid(seg_logits[:, 1:2])
        else:
            p_tumor = torch.sigmoid(seg_logits)
            
        x_masked = x * p_tumor
        
        # ------------------------------------------
        # 3. Forward Classification Network with ICFF
        # ------------------------------------------
        c0 = self.cls_init(x_masked)
        c1 = self.cls_stage1(c0) # (B, 2c, D/2, H/2, W/2)
        c1_fused = self.icff1(d2, c1)
        
        c2 = self.cls_stage2(c1_fused) # (B, 4c, D/4, H/4, W/4)
        c2_fused = self.icff2(d3, c2)
        
        c3 = self.cls_stage3(c2_fused) # (B, 8c, D/8, H/8, W/8)
        c3_fused = self.icff3(sb, c3)
        
        cls_logits = self.cls_head(c3_fused)
        
        return {
            'seg_logits': seg_logits,
            'cls_logits': cls_logits,
            'p_tumor': p_tumor
        }

if __name__ == '__main__':
    x = torch.randn(2, 1, 64, 64, 64)
    model = ResVNetFusion(num_seg_classes=2, base_channels=16)
    out = model(x)
    print("ResVNetFusion [Liver, Tumor] seg logits:", out['seg_logits'].shape)
    print("ResVNetFusion cls logits:", out['cls_logits'].shape)
    print("ResVNetFusion p_tumor:", out['p_tumor'].shape)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
