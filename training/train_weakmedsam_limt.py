"""
Training WeakMedSAM with Sub-Class Exploration (SCE) on the LiMT Dataset.
Adapts SAM ViT-B via SAMUS medical adapters and CNN cross-branch attention.
Jointly optimizes primary classification (Liver, Tumor) and sub-class exploration (K=4).
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add WeakMedSAM path
sys.path.append(os.path.abspath("ref/WeakMedSAM-main"))
from samus.build_sam_us import samus_model_registry
from dataset_weakmedsam import LiMTWeakMedSAMDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training WeakMedSAM (SCE) on LiMT using {device} ===")

    sam_ckpt = "C:/Users/hai49/.gemini/antigravity/scratch/cangio/results/models/sam_vit_b_01ec64.pth"
    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt_path = "checkpoints/best_weakmedsam_limt.pth"

    # Build SAMUS model
    parent_classes = 2 # Liver, Tumor
    child_classes = 4  # K=4 sub-classes
    print(f"Building SAMUS with parent_classes={parent_classes}, child_classes={child_classes}...")
    model = samus_model_registry["vit_b"](
        parent_classes=parent_classes,
        child_classes=child_classes,
        checkpoint=sam_ckpt
    ).to(device)

    # Datasets and Loaders
    train_ds = LiMTWeakMedSAMDataset(split="train", child_classes=child_classes, is_train=True)
    val_ds = LiMTWeakMedSAMDataset(split="val", child_classes=child_classes, is_train=False)

    batch_size = 4
    accum_steps = 4 # Effective batch size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    epochs = 5
    lr = 1e-4
    child_weight = 0.5

    # Trainable parameters are medical adapters and last blocks
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader) // accum_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_parent_loss = 0.0
        total_child_loss = 0.0
        correct_p = 0
        total_p = 0
        t0 = time.time()

        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")

        for step, batch in pbar:
            imgs = batch["img"].to(device, non_blocking=True)
            plab = batch["plab"].to(device, non_blocking=True)
            clab = batch["clab"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                p_out, c_out, cam = model(imgs)
                loss_p = F.binary_cross_entropy_with_logits(p_out, plab)
                loss_c = F.binary_cross_entropy_with_logits(c_out, clab)
                loss = (loss_p + child_weight * loss_c) / accum_steps

            scaler.scale(loss).backward()

            total_train_loss += loss.item() * accum_steps
            total_parent_loss += loss_p.item()
            total_child_loss += loss_c.item()

            preds = (torch.sigmoid(p_out) > 0.5).float()
            correct_p += (preds == plab).sum().item()
            total_p += plab.numel()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % 50 == 0:
                pbar.set_postfix({
                    "loss": f"{total_train_loss / (step + 1):.4f}",
                    "p_loss": f"{total_parent_loss / (step + 1):.4f}",
                    "c_loss": f"{total_child_loss / (step + 1):.4f}",
                    "acc": f"{correct_p / total_p * 100:.1f}%"
                })

        train_loss_avg = total_train_loss / len(train_loader)
        train_acc = correct_p / total_p * 100

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["img"].to(device, non_blocking=True)
                plab = batch["plab"].to(device, non_blocking=True)
                clab = batch["clab"].to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    p_out, c_out, cam = model(imgs)
                    loss_p = F.binary_cross_entropy_with_logits(p_out, plab)
                    loss_c = F.binary_cross_entropy_with_logits(c_out, clab)
                    v_loss = loss_p + child_weight * loss_c

                val_loss += v_loss.item()
                preds = (torch.sigmoid(p_out) > 0.5).float()
                val_correct += (preds == plab).sum().item()
                val_total += plab.numel()

        val_loss_avg = val_loss / len(val_loader)
        val_acc = val_correct / val_total * 100
        epoch_time = time.time() - t0

        print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) | Train Loss: {train_loss_avg:.4f} (Acc: {train_acc:.2f}%) | Val Loss: {val_loss_avg:.4f} (Acc: {val_acc:.2f}%)")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss_avg,
                "val_acc": val_acc
            }, best_ckpt_path)
            print(f"  --> Saved new best WeakMedSAM checkpoint to {best_ckpt_path} (Val Loss: {val_loss_avg:.4f})")

    print("\nWeakMedSAM training completed successfully.")

if __name__ == "__main__":
    main()
