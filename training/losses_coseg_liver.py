"""
Co-Seg Loss Function for Liver Multi-Task Dataset.
Implements the Two-Stage Collaborative Optimization Objective from Co-Seg (MICCAI 2025):
L_CoSeg = \lambda_prior * L_prior(coarse) + L_refined(refined) + \lambda_cls * L_cls + \lambda_burden * L_burden

Where:
- L_prior: Supervised loss on Forward 1 coarse liver & tumor logits (acting as prompt prior constraints).
- L_refined: Supervised loss on Forward 2 mutually guided liver & tumor logits + unified 6-class segmentation.
- L_cls: BCEWithLogitsLoss for 5-class multi-label lesion classification.
- L_burden: SmoothL1Loss for continuous tumor volume burden percentage regression.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_binary(pred_probs: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Soft Dice Loss for binary target (B, 1, D, H, W)."""
    p_flat = pred_probs.contiguous().view(pred_probs.shape[0], -1)
    t_flat = target.contiguous().view(target.shape[0], -1)
    
    intersection = (p_flat * t_flat).sum(dim=1)
    cardinality = p_flat.sum(dim=1) + t_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return (1.0 - dice).mean()


def dice_loss_multiclass(pred_probs: torch.Tensor, target_one_hot: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Soft Dice Loss across all classes (B, C, D, H, W)."""
    p_flat = pred_probs.contiguous().view(pred_probs.shape[0], pred_probs.shape[1], -1)
    t_flat = target_one_hot.contiguous().view(target_one_hot.shape[0], target_one_hot.shape[1], -1)
    
    intersection = (p_flat * t_flat).sum(dim=2)
    cardinality = p_flat.sum(dim=2) + t_flat.sum(dim=2)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    # Ignore background in dice penalty if desired, or average across all foreground classes
    return (1.0 - dice[:, 1:]).mean()


class CoSegMultiTaskLoss(nn.Module):
    """
    Unified Co-Seg loss combining:
    1. Forward 1 Prior Constraint Loss (Coarse Liver & Tumor)
    2. Forward 2 Mutually Refined Segmentation Loss (Refined Liver & Tumor + 6-class)
    3. Multi-label Classification Loss
    4. Tumor Burden Regression Loss
    """
    def __init__(
        self,
        lambda_prior: float = 0.5,
        lambda_tumor: float = 2.0,
        lambda_cls: float = 0.5,
        lambda_burden: float = 0.1,
        bce_cls_pos_weight: float = 2.0
    ):
        super().__init__()
        self.lambda_prior = lambda_prior
        self.lambda_tumor = lambda_tumor
        self.lambda_cls = lambda_cls
        self.lambda_burden = lambda_burden
        
        pos_weight = torch.tensor([bce_cls_pos_weight] * 5)
        self.cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.burden_criterion = nn.SmoothL1Loss()
        self.ce_seg = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        seg_targets: torch.Tensor,     # (B, D, H, W) in [0..5]
        cls_targets: torch.Tensor,     # (B, 5) float
        burden_targets: torch.Tensor   # (B, 1) float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        # Ground truths
        t_liver = (seg_targets >= 1).float().unsqueeze(1) # (B, 1, D, H, W)
        t_tumor = (seg_targets >= 2).float().unsqueeze(1) # (B, 1, D, H, W)
        
        # --- 1. Forward 1: Prior Constraint Loss ---
        coarse_liver = outputs['coarse_liver_logits']
        coarse_tumor = outputs['coarse_tumor_logits']
        
        loss_coarse_liver_bce = F.binary_cross_entropy_with_logits(coarse_liver, t_liver)
        loss_coarse_liver_dice = dice_loss_binary(torch.sigmoid(coarse_liver), t_liver)
        loss_coarse_liver = loss_coarse_liver_bce + loss_coarse_liver_dice
        
        loss_coarse_tumor_bce = F.binary_cross_entropy_with_logits(coarse_tumor, t_tumor)
        loss_coarse_tumor_dice = dice_loss_binary(torch.sigmoid(coarse_tumor), t_tumor)
        loss_coarse_tumor = loss_coarse_tumor_bce + loss_coarse_tumor_dice
        
        loss_prior = loss_coarse_liver + self.lambda_tumor * loss_coarse_tumor
        
        # --- 2. Forward 2: Refined Mutual Segmentation Loss ---
        refined_liver = outputs['refined_liver_logits']
        refined_tumor = outputs['refined_tumor_binary']
        seg_logits_6c = outputs['seg_logits'] # (B, 6, D, H, W)
        
        loss_ref_liver_bce = F.binary_cross_entropy_with_logits(refined_liver, t_liver)
        loss_ref_liver_dice = dice_loss_binary(torch.sigmoid(refined_liver), t_liver)
        loss_ref_liver = loss_ref_liver_bce + loss_ref_liver_dice
        
        loss_ref_tumor_bce = F.binary_cross_entropy_with_logits(refined_tumor, t_tumor)
        loss_ref_tumor_dice = dice_loss_binary(torch.sigmoid(refined_tumor), t_tumor)
        loss_ref_tumor = loss_ref_tumor_bce + loss_ref_tumor_dice
        
        # 6-class unified supervision (CE + Multi-class Dice)
        loss_ce_6c = self.ce_seg(seg_logits_6c, seg_targets)
        probs_6c = F.softmax(seg_logits_6c, dim=1)
        target_6c_onehot = F.one_hot(seg_targets, num_classes=6).permute(0, 4, 1, 2, 3).float()
        loss_dice_6c = dice_loss_multiclass(probs_6c, target_6c_onehot)
        loss_seg_6c = loss_ce_6c + loss_dice_6c
        
        loss_refined = loss_ref_liver + self.lambda_tumor * loss_ref_tumor + loss_seg_6c
        
        # --- 3. Multi-Task Heads Loss ---
        if self.cls_criterion.pos_weight.device != outputs['cls_logits'].device:
            self.cls_criterion.pos_weight = self.cls_criterion.pos_weight.to(outputs['cls_logits'].device)
            
        loss_cls = self.cls_criterion(outputs['cls_logits'], cls_targets)
        loss_burden = self.burden_criterion(outputs['tumor_burden'], burden_targets)
        
        # --- Total Objective ---
        total_loss = (
            self.lambda_prior * loss_prior +
            loss_refined +
            self.lambda_cls * loss_cls +
            self.lambda_burden * loss_burden
        )
        
        metrics_dict = {
            'loss_total': total_loss.item(),
            'loss_prior': loss_prior.item(),
            'loss_coarse_liver': loss_coarse_liver.item(),
            'loss_coarse_tumor': loss_coarse_tumor.item(),
            'loss_refined': loss_refined.item(),
            'loss_ref_liver': loss_ref_liver.item(),
            'loss_ref_tumor': loss_ref_tumor.item(),
            'loss_seg_6c': loss_seg_6c.item(),
            'loss_cls': loss_cls.item(),
            'loss_burden': loss_burden.item()
        }
        
        return total_loss, metrics_dict
