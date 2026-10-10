import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import LiMTMultiTaskDataset
from models.resvnet_fusion import ResVNetFusion
from losses import BinaryTumorDiceLoss, FocalLoss3D
from metrics import evaluate_segmentation_batch, evaluate_classification_batch

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class EnhancedRef78Loss(torch.nn.Module):
    """
    Ref [78] Multi-task loss upgraded with Tumor Focal and Binary Dice Loss
    to tackle the severe class imbalance in LiMT.
    """
    def __init__(self, alpha=0.7, tumor_weight=3.5, focal_gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.tumor_weight = tumor_weight
        self.dice = BinaryTumorDiceLoss()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.focal_gamma = focal_gamma

    def focal_binary_loss(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal = ((1.0 - pt) ** self.focal_gamma) * bce
        return torch.mean(focal)

    def forward(self, outputs, seg_targets, cls_targets):
        losses = {}
        total_loss = 0.0
        
        if 'seg_logits' in outputs and seg_targets is not None:
            seg_logits = outputs['seg_logits']
            
            # Liver: channel 0
            liver_targets = (seg_targets >= 1).unsqueeze(1).float()
            # Tumor: channel 1
            tumor_targets = (seg_targets >= 2).unsqueeze(1).float()
            
            # Liver loss
            l_liver_bce = self.bce(seg_logits[:, 0:1], liver_targets)
            p_liver = torch.sigmoid(seg_logits[:, 0:1])
            l_liver_dice = 1.0 - (2.0 * (p_liver * liver_targets).sum() + 1e-5) / (p_liver.sum() + liver_targets.sum() + 1e-5)
            l_liver = l_liver_bce + l_liver_dice
            
            # Tumor loss (Dice + Focal)
            p_tumor = torch.sigmoid(seg_logits[:, 1:2])
            l_tumor_dice = 1.0 - (2.0 * (p_tumor * tumor_targets).sum() + 1e-5) / (p_tumor.sum() + tumor_targets.sum() + 1e-5)
            l_tumor_focal = self.focal_binary_loss(seg_logits[:, 1:2], tumor_targets)
            l_tumor = l_tumor_dice + 2.0 * l_tumor_focal
            
            l_seg = 0.5 * l_liver + self.tumor_weight * l_tumor
            losses['loss_liver'] = l_liver.item()
            losses['loss_tumor'] = l_tumor.item()
            losses['loss_seg'] = l_seg.item()
            total_loss += self.alpha * l_seg
            
        if 'cls_logits' in outputs and cls_targets is not None:
            cls_logits = outputs['cls_logits']
            l_cls = self.bce(cls_logits, cls_targets)
            losses['loss_cls'] = l_cls.item()
            total_loss += (1.0 - self.alpha) * l_cls
            
        losses['loss_total'] = total_loss.item()
        return total_loss, losses

import torch.nn.functional as F

def main():
    set_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    target_shape = (80, 80, 80)
    train_ds = LiMTMultiTaskDataset(
        data_dir="LiMT Dataset", splits_csv="LiMT Dataset/limt_splits.csv",
        split_type="3way", split_val="train",
        target_shape=target_shape, hu_min=-200.0, hu_max=340.0,
        use_clahe=True, augment=True, cache_data=True
    )
    val_ds = LiMTMultiTaskDataset(
        data_dir="LiMT Dataset", splits_csv="LiMT Dataset/limt_splits.csv",
        split_type="3way", split_val="val",
        target_shape=target_shape, hu_min=-200.0, hu_max=340.0,
        use_clahe=True, augment=False, cache_data=True
    )
    test_ds = LiMTMultiTaskDataset(
        data_dir="LiMT Dataset", splits_csv="LiMT Dataset/limt_splits.csv",
        split_type="3way", split_val="test",
        target_shape=target_shape, hu_min=-200.0, hu_max=340.0,
        use_clahe=True, augment=False, cache_data=True
    )
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)
    
    model = ResVNetFusion(in_channels=1, num_seg_classes=2, num_cls_classes=5, base_channels=16).to(device)
    criterion = EnhancedRef78Loss(alpha=0.7, tumor_weight=3.5).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 8
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_score = -float('inf')
    best_ckpt_path = "checkpoints/best_ref78_enhanced.pth"
    
    print(f"\n--- Training Ref [78] ResVNetFusion with Enhanced Tumor Loss ({epochs} Epochs) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            seg_targets = batch['segmentation'].to(device)
            cls_targets = batch['classification'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss, _ = criterion(outputs, seg_targets, cls_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        seg_dices_liver, seg_dices_tumor = [], []
        cls_preds, cls_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                seg_targets = batch['segmentation'].to(device)
                outputs = model(images)
                res = evaluate_segmentation_batch(outputs['seg_logits'], seg_targets, tumor_thresh=0.5)
                seg_dices_liver.append(res['liver_dice'])
                seg_dices_tumor.append(res['tumor_dice'])
                cls_preds.append(outputs['cls_logits'].cpu())
                cls_targets.append(batch['classification'].cpu())
                
        l_d = np.mean(seg_dices_liver)
        t_d = np.mean(seg_dices_tumor)
        c_res = evaluate_classification_batch(torch.cat(cls_preds, 0), torch.cat(cls_targets, 0))
        
        score = t_d * 3.0 + l_d + c_res['f1']
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/len(train_loader):.3f} | "
              f"Liver Dice: {l_d:.1f}% | Tumor Dice: {t_d:.1f}% | Cls Acc: {c_res['hamming_acc']:.1f}% | Cls F1: {c_res['f1']:.1f}%")
              
        if score > best_score:
            best_score = score
            torch.save({'model_state_dict': model.state_dict(), 'val_metrics': {'liver_dice': l_d, 'tumor_dice': t_d}}, best_ckpt_path)
            
    print(f"\nBest model saved to {best_ckpt_path}")
    
    # Test evaluation
    print("\n--- Final Evaluation on Test Split ---")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()
    
    for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
        seg_dices_liver, seg_dices_tumor = [], []
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                seg_targets = batch['segmentation'].to(device)
                outputs = model(images)
                res = evaluate_segmentation_batch(outputs['seg_logits'], seg_targets, tumor_thresh=th)
                seg_dices_liver.append(res['liver_dice'])
                seg_dices_tumor.append(res['tumor_dice'])
        print(f"Threshold {th:.2f} -> Liver Dice: {np.mean(seg_dices_liver):.2f}% | Tumor Dice: {np.mean(seg_dices_tumor):.2f}%")

if __name__ == '__main__':
    main()
