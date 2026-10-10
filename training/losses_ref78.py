import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDiceLoss(nn.Module):
    """Smooth binary Dice Loss."""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets_f = targets.float()
        
        dims = (1, 2, 3, 4) if probs.dim() == 5 else (1, 2, 3)
        intersection = torch.sum(probs * targets_f, dim=dims)
        cardinality = torch.sum(probs + targets_f, dim=dims)
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        has_pos = (torch.sum(targets_f, dim=dims) > 0).float()
        loss = has_pos * (1.0 - dice) + (1.0 - has_pos) * torch.mean(probs, dim=dims)
        return torch.mean(loss)

class Ref78MultiTaskLoss(nn.Module):
    """
    Exact Loss Formulation from Chen et al. (Ref [78]):
    JointLoss = alpha * DiceLoss_Seg + (1 - alpha) * CELoss_Cls
    Extended for multi-task liver & tumor segmentation:
      L_seg = L_tumor_dice + 0.5 * L_liver_dice + L_bce
      L_cls = BCEWithLogitsLoss
    """
    def __init__(self, alpha=0.7, tumor_weight=2.0):
        super().__init__()
        self.alpha = alpha
        self.tumor_weight = tumor_weight
        self.dice = BinaryDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, seg_targets, cls_targets):
        losses = {}
        total_loss = 0.0
        
        if 'seg_logits' in outputs and seg_targets is not None:
            seg_logits = outputs['seg_logits']
            
            if seg_logits.shape[1] == 2:
                # Channel 0: Liver (Y >= 1)
                # Channel 1: Tumor (Y >= 2)
                liver_targets = (seg_targets >= 1).unsqueeze(1).float()
                tumor_targets = (seg_targets >= 2).unsqueeze(1).float()
                
                l_liver_dice = self.dice(seg_logits[:, 0:1], liver_targets)
                l_liver_bce = self.bce(seg_logits[:, 0:1], liver_targets)
                l_liver = l_liver_dice + l_liver_bce
                
                l_tumor_dice = self.dice(seg_logits[:, 1:2], tumor_targets)
                l_tumor_bce = self.bce(seg_logits[:, 1:2], tumor_targets)
                l_tumor = l_tumor_dice + l_tumor_bce
                
                l_seg = 0.5 * l_liver + self.tumor_weight * l_tumor
                losses['loss_liver_dice'] = l_liver_dice.item()
                losses['loss_tumor_dice'] = l_tumor_dice.item()
            else:
                tumor_targets = (seg_targets >= 2).unsqueeze(1).float()
                l_tumor_dice = self.dice(seg_logits, tumor_targets)
                l_tumor_bce = self.bce(seg_logits, tumor_targets)
                l_seg = l_tumor_dice + l_tumor_bce
                losses['loss_tumor_dice'] = l_tumor_dice.item()
                
            losses['loss_seg'] = l_seg.item()
            total_loss = total_loss + self.alpha * l_seg
            
        if 'cls_logits' in outputs and cls_targets is not None:
            cls_logits = outputs['cls_logits']
            l_cls = self.bce(cls_logits, cls_targets)
            losses['loss_cls'] = l_cls.item()
            total_loss = total_loss + (1.0 - self.alpha) * l_cls
            
        losses['loss_total'] = total_loss.item()
        return total_loss, losses

# Also keep existing loss classes for backward compatibility
from losses import MultiClassDiceLoss, BinaryTumorDiceLoss, FocalLoss3D, TumorAwareMultiTaskLoss, MultiTaskLoss
