"""
Multi-Task Loss for Combined Co-Seg + SOPSeg Architecture on LiMT Dataset.
Combines:
1. Co-Seg Collaborative Prior Loss on coarse logits (Forward 1)
2. SOPSeg Multi-Scale Progressive Edge and Mask Supervision (16^3, 32^3, 64^3)
3. Refined Segmentation Loss (Dice + CE on 6-class unified map)
4. Multi-Label Classification Loss
5. Continuous Tumor Burden Regression Loss
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses_coseg_liver import dice_loss_binary, dice_loss_multiclass


def extract_3d_edges(binary_mask: torch.Tensor) -> torch.Tensor:
    """Extracts 3D boundary edge map using morphological dilation - erosion."""
    dilated = F.max_pool3d(binary_mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool3d(-binary_mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


class CoSegSOPSegLoss(nn.Module):
    def __init__(
        self,
        lambda_prior: float = 0.5,
        lambda_tumor: float = 2.5,
        lambda_edge: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_burden: float = 0.1,
        bce_cls_pos_weight: float = 2.0
    ):
        super().__init__()
        self.lambda_prior = lambda_prior
        self.lambda_tumor = lambda_tumor
        self.lambda_edge = lambda_edge
        self.lambda_cls = lambda_cls
        self.lambda_burden = lambda_burden
        
        pos_weight = torch.tensor([bce_cls_pos_weight] * 5)
        self.cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.burden_criterion = nn.SmoothL1Loss()
        self.ce_seg = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        seg_targets: torch.Tensor,     # (B, D, H, W)
        cls_targets: torch.Tensor,     # (B, 5)
        burden_targets: torch.Tensor   # (B, 1)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        # Ground truths at full resolution (64^3)
        t_liver = (seg_targets >= 1).float().unsqueeze(1) # (B, 1, 64, 64, 64)
        t_tumor = (seg_targets >= 2).float().unsqueeze(1) # (B, 1, 64, 64, 64)
        
        t_edge_liver = extract_3d_edges(t_liver)
        t_edge_tumor = extract_3d_edges(t_tumor)
        
        # 1. Co-Seg Prior Loss on Forward 1 coarse predictions
        coarse_liv = outputs['coarse_liver_logits']
        coarse_tum = outputs['coarse_tumor_logits']
        
        loss_prior_liv = (
            F.binary_cross_entropy_with_logits(coarse_liv, t_liver) +
            dice_loss_binary(torch.sigmoid(coarse_liv), t_liver)
        )
        loss_prior_tum = (
            F.binary_cross_entropy_with_logits(coarse_tum, t_tumor) +
            dice_loss_binary(torch.sigmoid(coarse_tum), t_tumor)
        )
        loss_prior = loss_prior_liv + self.lambda_tumor * loss_prior_tum
        
        # 2. SOPSeg Multi-Scale Progressive Edge & Mask Refinement Loss
        loss_prm = 0.0
        scales = [
            ('p16_liver', 'p16_tumor', 16, 0.25),
            ('p32_liver', 'p32_tumor', 32, 0.5),
            ('refined_liver_mask', 'refined_tumor_mask', 64, 1.0)
        ]
        
        for k_liv, k_tum, s, w_scale in scales:
            if s == 64:
                m_liv = outputs['refined_liver_mask']
                e_liv = outputs['refined_liver_edge']
                m_tum = outputs['refined_tumor_mask']
                e_tum = outputs['refined_tumor_edge']
            else:
                p_liv = outputs[k_liv]
                m_liv = p_liv[:, 0:1]
                e_liv = p_liv[:, 1:2]
                
                p_tum = outputs[k_tum]
                m_tum = p_tum[:, 0:1]
                e_tum = p_tum[:, 1:2]
                
            # Interpolate ground truth to current scale
            gt_m_liv = F.interpolate(t_liver, size=(s, s, s), mode='nearest')
            gt_e_liv = F.interpolate(t_edge_liver, size=(s, s, s), mode='nearest')
            gt_m_tum = F.interpolate(t_tumor, size=(s, s, s), mode='nearest')
            gt_e_tum = F.interpolate(t_edge_tumor, size=(s, s, s), mode='nearest')
            
            # Mask losses
            l_m_liv = F.binary_cross_entropy_with_logits(m_liv, gt_m_liv) + dice_loss_binary(torch.sigmoid(m_liv), gt_m_liv)
            l_m_tum = F.binary_cross_entropy_with_logits(m_tum, gt_m_tum) + dice_loss_binary(torch.sigmoid(m_tum), gt_m_tum)
            
            # Edge losses (SOPSeg explicit edge guidance)
            l_e_liv = F.binary_cross_entropy_with_logits(e_liv, gt_e_liv) + dice_loss_binary(torch.sigmoid(e_liv), gt_e_liv)
            l_e_tum = F.binary_cross_entropy_with_logits(e_tum, gt_e_tum) + dice_loss_binary(torch.sigmoid(e_tum), gt_e_tum)
            
            scale_loss = (l_m_liv + self.lambda_tumor * l_m_tum) + self.lambda_edge * (l_e_liv + self.lambda_tumor * l_e_tum)
            loss_prm += w_scale * scale_loss
            
        # 3. Unified 6-Class Segmentation Loss
        seg_6c = outputs['seg_logits']
        loss_ce_6c = self.ce_seg(seg_6c, seg_targets)
        probs_6c = F.softmax(seg_6c, dim=1)
        target_6c_onehot = F.one_hot(seg_targets, num_classes=6).permute(0, 4, 1, 2, 3).float()
        loss_dice_6c = dice_loss_multiclass(probs_6c, target_6c_onehot)
        loss_seg_6c = loss_ce_6c + loss_dice_6c
        
        # 4. Multi-Task Classification & Burden Losses
        if self.cls_criterion.pos_weight.device != outputs['cls_logits'].device:
            self.cls_criterion.pos_weight = self.cls_criterion.pos_weight.to(outputs['cls_logits'].device)
            
        loss_cls = self.cls_criterion(outputs['cls_logits'], cls_targets)
        loss_burden = self.burden_criterion(outputs['tumor_burden'], burden_targets)
        
        # Total Loss
        total_loss = (
            self.lambda_prior * loss_prior +
            loss_prm +
            loss_seg_6c +
            self.lambda_cls * loss_cls +
            self.lambda_burden * loss_burden
        )
        
        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_prior': loss_prior.item(),
            'loss_prm': loss_prm.item(),
            'loss_seg_6c': loss_seg_6c.item(),
            'loss_cls': loss_cls.item(),
            'loss_burden': loss_burden.item()
        }
