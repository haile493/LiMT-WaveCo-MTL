"""
Multi-Task Loss for WeedSense on Liver CT (LiMT Dataset).
Adheres to the WeedSense loss balancing protocol:
- Main Segmentation: Weighted CE + MultiClassDice + BinaryTumorDice + Focal Loss
- Deep Auxiliary Supervision: 4 auxiliary heads weighted by 0.4 each (matching WeedSense auxiliary weights [1.0, 0.4, 0.4, 0.4, 0.4])
- Multi-label Classification: BCEWithLogitsLoss
- Quantitative Burden Regression: Smooth L1 / MSE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import MultiClassDiceLoss, BinaryTumorDiceLoss, FocalLoss3D


class WeedSenseLiverMultiTaskLoss(nn.Module):
    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_burden: float = 0.5,
        aux_weight: float = 0.4,
        beta_tumor: float = 2.0,
        device = None
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls
        self.lambda_burden = lambda_burden
        self.aux_weight = aux_weight
        self.beta_tumor = beta_tumor

        # Class weights for liver & lesion voxels
        class_weights = torch.tensor([0.1, 0.5, 4.0, 4.0, 4.0, 4.0], dtype=torch.float32)
        if device is not None:
            class_weights = class_weights.to(device)
        self.register_buffer("class_weights", class_weights)

        self.dice_loss = MultiClassDiceLoss()
        self.tumor_dice_loss = BinaryTumorDiceLoss()
        self.focal_loss = FocalLoss3D(gamma=2.0, weight=self.class_weights)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = nn.SmoothL1Loss()

    def compute_seg_loss(self, logits, targets):
        l_ce = F.cross_entropy(logits, targets, weight=self.class_weights)
        l_mc_dice = self.dice_loss(logits, targets)
        l_tumor_dice = self.tumor_dice_loss(logits, targets)
        l_focal = self.focal_loss(logits, targets)
        return l_ce + l_mc_dice + self.beta_tumor * l_tumor_dice + 0.5 * l_focal

    def forward(self, outputs, seg_targets, cls_targets, burden_targets=None):
        losses = {}
        total_loss = 0.0

        # 1. Main Segmentation
        if 'seg_logits' in outputs and seg_targets is not None:
            l_main_seg = self.compute_seg_loss(outputs['seg_logits'], seg_targets)
            losses['loss_main_seg'] = l_main_seg.item()
            seg_total = l_main_seg

            # 2. Auxiliary Deep Supervision (Stem, S3, S4, S5)
            aux_keys = ['aux_stem', 'aux_s3', 'aux_s4', 'aux_s5']
            aux_losses = []
            for k in aux_keys:
                if k in outputs and outputs[k] is not None:
                    l_aux_k = self.dice_loss(outputs[k], seg_targets) + self.tumor_dice_loss(outputs[k], seg_targets)
                    aux_losses.append(l_aux_k)

            if len(aux_losses) > 0:
                l_aux_total = sum(aux_losses) * self.aux_weight
                seg_total = seg_total + l_aux_total
                losses['loss_aux'] = l_aux_total.item()

            losses['loss_seg'] = seg_total.item()
            total_loss = total_loss + self.lambda_seg * seg_total

        # 3. Multi-label Lesion Classification
        if 'cls_logits' in outputs and cls_targets is not None:
            l_cls = self.bce_loss(outputs['cls_logits'], cls_targets)
            losses['loss_cls'] = l_cls.item()
            total_loss = total_loss + self.lambda_cls * l_cls

        # 4. Quantitative Tumor Burden Regression
        if 'tumor_burden' in outputs and burden_targets is not None:
            # Burden is in percent (0 - 100), scale to [0, 1] for balanced gradient scaling
            pred_b = outputs['tumor_burden'].squeeze(-1) / 100.0
            targ_b = burden_targets.squeeze(-1) / 100.0
            l_burden = self.reg_loss(pred_b, targ_b) * 10.0
            losses['loss_burden'] = l_burden.item()
            total_loss = total_loss + self.lambda_burden * l_burden

        losses['loss_total'] = total_loss.item()
        return total_loss, losses
