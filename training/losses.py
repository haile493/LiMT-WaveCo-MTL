import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    """
    Multi-Class Generalized Dice Loss.
    Computes smooth Dice coefficient per class and averages across present classes.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Exclude background class (index 0) from dice loss average to focus on liver & lesions
        dice_loss = 1.0 - torch.mean(dice_score[1:])
        return dice_loss

class BinaryTumorDiceLoss(nn.Module):
    """
    Dedicated Binary Dice Loss for all tumor lesions combined (classes 2, 3, 4, 5).
    Crucial for combating severe class imbalance between liver and small lesions.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        # Sum predicted probabilities of all tumor classes (indices 2..5)
        pred_tumor_prob = torch.sum(probs[:, 2:], dim=1) # (B, D, H, W)
        true_tumor_mask = (targets >= 2).float()          # (B, D, H, W)
        
        dims = (1, 2, 3)
        intersection = torch.sum(pred_tumor_prob * true_tumor_mask, dim=dims)
        cardinality = torch.sum(pred_tumor_prob + true_tumor_mask, dim=dims)
        
        # Batch-wise dice
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # If a case has no tumors in ground truth, only penalize false positives
        has_tumor = (torch.sum(true_tumor_mask, dim=dims) > 0).float()
        # For tumor cases: 1 - dice
        # For non-tumor cases: pred_tumor_prob mean
        case_loss = has_tumor * (1.0 - dice) + (1.0 - has_tumor) * torch.mean(pred_tumor_prob, dim=dims)
        return torch.mean(case_loss)

class FocalLoss3D(nn.Module):
    """3D Multi-Class Focal Loss for hard lesion voxels."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)

class TumorAwareMultiTaskLoss(nn.Module):
    """
    Advanced Multi-Task Loss with Tumor-Targeted Segmentation Objective:
    L_total = L_seg + lambda_cls * L_cls
    where:
      L_seg = L_ce_weighted + L_multiclass_dice + beta_tumor * L_tumor_dice + L_focal
      L_cls = L_bce
    """
    def __init__(
        self,
        lambda_seg=1.0,
        lambda_cls=0.5,
        beta_tumor=2.0,
        cls_pos_weight=None,
        device=None
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls
        self.beta_tumor = beta_tumor
        
        # Heavy penalty on lesion classes [Background: 0.1, Liver: 0.5, Cyst: 4.0, Hemangioma: 4.0, Met: 4.0, HCC: 4.0]
        class_weights = torch.tensor([0.1, 0.5, 4.0, 4.0, 4.0, 4.0], dtype=torch.float32)
        if device is not None:
            class_weights = class_weights.to(device)
        self.register_buffer("class_weights", class_weights)
        
        self.dice_loss = MultiClassDiceLoss()
        self.tumor_dice_loss = BinaryTumorDiceLoss()
        self.focal_loss = FocalLoss3D(gamma=2.0, weight=self.class_weights)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight)

    def forward(self, outputs, seg_targets, cls_targets):
        losses = {}
        total_loss = 0.0
        
        if 'seg_logits' in outputs and seg_targets is not None:
            seg_logits = outputs['seg_logits']
            
            l_ce = F.cross_entropy(seg_logits, seg_targets, weight=self.class_weights)
            l_mc_dice = self.dice_loss(seg_logits, seg_targets)
            l_tumor_dice = self.tumor_dice_loss(seg_logits, seg_targets)
            l_focal = self.focal_loss(seg_logits, seg_targets)
            
            l_seg = l_ce + l_mc_dice + self.beta_tumor * l_tumor_dice + 0.5 * l_focal
            
            # Deep supervision support
            if 'aux_seg_logits' in outputs and outputs['aux_seg_logits'] is not None:
                aux_logits = outputs['aux_seg_logits']
                l_aux = self.tumor_dice_loss(aux_logits, seg_targets) + self.dice_loss(aux_logits, seg_targets)
                l_seg = l_seg + 0.5 * l_aux
                losses['loss_aux'] = l_aux.item()
                
            losses['loss_ce'] = l_ce.item()
            losses['loss_mc_dice'] = l_mc_dice.item()
            losses['loss_tumor_dice'] = l_tumor_dice.item()
            losses['loss_seg'] = l_seg.item()
            total_loss = total_loss + self.lambda_seg * l_seg
            
        if 'cls_logits' in outputs and cls_targets is not None:
            cls_logits = outputs['cls_logits']
            l_bce = self.bce_loss(cls_logits, cls_targets)
            losses['loss_bce'] = l_bce.item()
            losses['loss_cls'] = l_bce.item()
            total_loss = total_loss + self.lambda_cls * l_bce
            
        losses['loss_total'] = total_loss.item()
        return total_loss, losses

# Backward compatible MultiTaskLoss class
class MultiTaskLoss(TumorAwareMultiTaskLoss):
    pass

if __name__ == '__main__':
    criterion = TumorAwareMultiTaskLoss()
    seg_logits = torch.randn(2, 6, 32, 32, 32)
    cls_logits = torch.randn(2, 5)
    seg_targets = torch.randint(0, 6, (2, 32, 32, 32))
    cls_targets = torch.randint(0, 2, (2, 5)).float()
    
    loss, loss_dict = criterion({'seg_logits': seg_logits, 'cls_logits': cls_logits}, seg_targets, cls_targets)
    print("Computed total loss:", loss.item())
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
