"""
Training script for WeakMedSAM-Seg on LiMT dataset.
High-resolution 2D multi-task segmentation with Attention U-Net and ImageNet-pretrained encoder.
Optimizes combined Dice + Focal + BCE losses with anatomical prior hierarchy.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset_weakmedsam import LiMTWeakMedSAMDataset
from models.weakmedsam_seg_net import WeakMedSAM_SegNet

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2.0 * torch.sum(probs * targets, dim=(2, 3)) + self.smooth
        den = torch.sum(probs, dim=(2, 3)) + torch.sum(targets, dim=(2, 3)) + self.smooth
        dice = num / den
        return 1.0 - torch.mean(dice)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_loss = alpha_factor * ((1.0 - p_t) ** self.gamma) * bce
        return torch.mean(focal_loss)

def compute_batch_dice(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    num = 2.0 * torch.sum(preds * targets, dim=(2, 3))
    den = torch.sum(preds, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))
    # Where target has pixels
    mask = (den > 0)
    if mask.sum() == 0:
        return 1.0
    return (num[mask] / den[mask]).mean().item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training WeakMedSAM-Seg on LiMT ({device}) ===")

    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt_path = "checkpoints/best_weakmedsam_seg_limt.pth"

    # Datasets
    train_ds = LiMTWeakMedSAMDataset(split="train", is_train=True)
    val_ds = LiMTWeakMedSAMDataset(split="val", is_train=False)

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = WeakMedSAM_SegNet(pretrained=True).to(device)

    dice_loss_fn = SoftDiceLoss()
    focal_loss_fn = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    epochs = 12
    lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_composite_score = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        liver_loss_tot = 0.0
        tumor_loss_tot = 0.0
        t0 = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")

        for step, batch in pbar:
            imgs = batch["img"].to(device, non_blocking=True)
            seg_target = batch["seg"].to(device, non_blocking=True) # [B, 2, 256, 256]
            plab_target = batch["plab"].to(device, non_blocking=True) # [B, 2]

            target_liver = seg_target[:, 0:1]
            target_tumor = seg_target[:, 1:2]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                out = model(imgs)
                seg_logits = out["seg_logits"]
                cls_logits = out["cls_logits"]

                logit_liver = seg_logits[:, 0:1]
                logit_tumor = seg_logits[:, 1:2]

                # Liver loss: BCE + Dice
                l_liv_bce = bce_loss_fn(logit_liver, target_liver)
                l_liv_dice = dice_loss_fn(logit_liver, target_liver)
                loss_liver = l_liv_bce + l_liv_dice

                # Tumor loss: Focal + 2.0 * Dice
                l_tum_focal = focal_loss_fn(logit_tumor, target_tumor)
                l_tum_dice = dice_loss_fn(logit_tumor, target_tumor)
                loss_tumor = l_tum_focal + 2.0 * l_tum_dice

                # Classification auxiliary loss
                loss_cls = bce_loss_fn(cls_logits, plab_target)

                # Total loss
                loss = loss_liver + 1.5 * loss_tumor + 0.2 * loss_cls

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            liver_loss_tot += loss_liver.item()
            tumor_loss_tot += loss_tumor.item()

            if (step + 1) % 20 == 0:
                pbar.set_postfix({
                    "loss": f"{total_loss / (step + 1):.4f}",
                    "liv_loss": f"{liver_loss_tot / (step + 1):.4f}",
                    "tum_loss": f"{tumor_loss_tot / (step + 1):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
                })

        # Validation
        model.eval()
        val_liver_dices = []
        val_tumor_dices = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["img"].to(device, non_blocking=True)
                seg_target = batch["seg"].to(device, non_blocking=True)

                target_liver = seg_target[:, 0:1]
                target_tumor = seg_target[:, 1:2]

                with torch.amp.autocast('cuda'):
                    out = model(imgs)
                    seg_logits = out["seg_logits"]
                    logit_liver = seg_logits[:, 0:1]
                    logit_tumor = seg_logits[:, 1:2]

                # Compute Dice on positive slices
                l_dice = compute_batch_dice(logit_liver, target_liver)
                val_liver_dices.append(l_dice)

                # Only evaluate tumor dice on slices with ground truth tumor
                has_tumor = (target_tumor.sum(dim=(2, 3)) > 0).squeeze(1)
                if has_tumor.sum() > 0:
                    t_dice = compute_batch_dice(logit_tumor[has_tumor], target_tumor[has_tumor])
                    val_tumor_dices.append(t_dice)

        mean_val_liver_dice = np.mean(val_liver_dices) * 100.0
        mean_val_tumor_dice = np.mean(val_tumor_dices) * 100.0 if len(val_tumor_dices) > 0 else 0.0
        composite_score = mean_val_liver_dice + 1.5 * mean_val_tumor_dice
        epoch_time = time.time() - t0

        print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) | Val Liver Dice: {mean_val_liver_dice:.2f}% | Val Tumor Dice: {mean_val_tumor_dice:.2f}% | Composite: {composite_score:.2f}")

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_liver_dice": mean_val_liver_dice,
                "val_tumor_dice": mean_val_tumor_dice,
                "composite_score": composite_score
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path}!")

    print("\nTraining completed successfully.")

if __name__ == "__main__":
    main()
