"""
Training and Evaluation script for WeedSense applied to the Liver Multi-Task (LiMT) Dataset.
Executes:
1. Training WeedSenseLiverNet with 3D Dual-path UIB Encoder + BGA + Bifurcated Decoder + Aux heads
2. Validation after each epoch and saving best model checkpoint
3. Final evaluation on the 30 held-out test cases
4. Quantitative comparison against baseline models
"""

import os
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

from dataset import LiMTMultiTaskDataset
from models.weedsense_liver import WeedSenseLiverNet
from losses_weedsense_liver import WeedSenseLiverMultiTaskLoss
from metrics import evaluate_segmentation_batch

CLASS_NAMES = ["Liver (0)", "Cyst (1)", "Hemangioma (2)", "Metastasis (3)", "HCC (4)"]


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss_accum = 0.0
    seg_loss_accum = 0.0
    cls_loss_accum = 0.0
    burden_loss_accum = 0.0
    aux_loss_accum = 0.0

    for batch in dataloader:
        images = batch['image'].to(device)
        seg_targets = batch['segmentation'].to(device)
        cls_targets = batch['classification'].to(device)
        burden_targets = batch['tumor_burden'].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss, loss_dict = criterion(outputs, seg_targets, cls_targets, burden_targets)

        loss.backward()
        optimizer.step()

        total_loss_accum += loss.item()
        seg_loss_accum += loss_dict.get('loss_main_seg', 0.0)
        cls_loss_accum += loss_dict.get('loss_cls', 0.0)
        burden_loss_accum += loss_dict.get('loss_burden', 0.0)
        aux_loss_accum += loss_dict.get('loss_aux', 0.0)

    n = len(dataloader)
    return {
        'total_loss': total_loss_accum / n,
        'seg_loss': seg_loss_accum / n,
        'cls_loss': cls_loss_accum / n,
        'burden_loss': burden_loss_accum / n,
        'aux_loss': aux_loss_accum / n
    }


@torch.no_grad()
def evaluate_split(model, dataloader, criterion, device):
    model.eval()
    total_loss_accum = 0.0
    seg_dices_liver = []
    seg_dices_tumor = []
    per_class_dices = {i: [] for i in range(1, 6)}

    cls_preds_all = []
    cls_targets_all = []
    burden_preds_all = []
    burden_targets_all = []

    for batch in dataloader:
        images = batch['image'].to(device)
        seg_targets = batch['segmentation'].to(device)
        cls_targets = batch['classification'].to(device)
        burden_targets = batch['tumor_burden'].to(device)

        outputs = model(images)
        loss, _ = criterion(outputs, seg_targets, cls_targets, burden_targets)
        total_loss_accum += loss.item()

        # Segmentation metrics
        seg_logits = outputs['seg_logits']
        seg_preds = torch.argmax(seg_logits, dim=1)

        batch_seg_metrics = evaluate_segmentation_batch(seg_logits, seg_targets)
        seg_dices_liver.append(batch_seg_metrics['liver_dice'])
        seg_dices_tumor.append(batch_seg_metrics['tumor_dice'])

        # Per-class Dice
        for c in range(1, 6):
            pred_c = (seg_preds == c).float()
            true_c = (seg_targets == c).float()
            for b in range(images.shape[0]):
                inter = (pred_c[b] * true_c[b]).sum().item()
                card = pred_c[b].sum().item() + true_c[b].sum().item()
                if card > 0:
                    per_class_dices[c].append((2.0 * inter) / card)

        # Classification metrics
        cls_probs = torch.sigmoid(outputs['cls_logits']).cpu().numpy()
        cls_preds_all.append(cls_probs)
        cls_targets_all.append(cls_targets.cpu().numpy())

        # Burden regression metrics
        b_preds = outputs['tumor_burden'].cpu().numpy().flatten()
        b_targets = burden_targets.cpu().numpy().flatten()
        burden_preds_all.extend(b_preds)
        burden_targets_all.extend(b_targets)

    # Aggregate
    cls_probs_arr = np.concatenate(cls_preds_all, axis=0)
    cls_targets_arr = np.concatenate(cls_targets_all, axis=0)
    cls_binary_preds = (cls_probs_arr >= 0.5).astype(int)

    hamming_acc = float(accuracy_score(cls_targets_arr, cls_binary_preds) * 100)
    subset_acc = float(np.mean(np.all(cls_targets_arr == cls_binary_preds, axis=1)) * 100)
    macro_f1 = float(f1_score(cls_targets_arr, cls_binary_preds, average='macro', zero_division=0) * 100)
    micro_f1 = float(f1_score(cls_targets_arr, cls_binary_preds, average='micro', zero_division=0) * 100)

    # Burden Regression
    bp = np.array(burden_preds_all)
    bt = np.array(burden_targets_all)
    b_mae = float(mean_absolute_error(bt, bp))
    b_rmse = float(np.sqrt(mean_squared_error(bt, bp)))
    b_r2 = float(r2_score(bt, bp)) if len(np.unique(bt)) > 1 else 0.0

    return {
        'loss': total_loss_accum / len(dataloader),
        'liver_dice': float(np.mean(seg_dices_liver) * 100),
        'tumor_dice': float(np.mean(seg_dices_tumor) * 100),
        'per_class_dice': {c: float(np.mean(v) * 100) if len(v) > 0 else 0.0 for c, v in per_class_dices.items()},
        'hamming_acc': hamming_acc,
        'subset_acc': subset_acc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'burden_mae': b_mae,
        'burden_rmse': b_rmse,
        'burden_r2': b_r2
    }


def main():
    set_seed(1337)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Datasets
    print("\n--- Loading LiMT Dataset (3-Way Split) ---")
    train_ds = LiMTMultiTaskDataset(split_type="3way", split_val="train", target_shape=(64, 64, 64), use_clahe=True, augment=True, cache_data=True)
    val_ds = LiMTMultiTaskDataset(split_type="3way", split_val="val", target_shape=(64, 64, 64), use_clahe=True, augment=False, cache_data=True)
    test_ds = LiMTMultiTaskDataset(split_type="3way", split_val="test", target_shape=(64, 64, 64), use_clahe=True, augment=False, cache_data=True)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # Initialize WeedSense Liver Network
    model = WeedSenseLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized WeedSenseLiverNet: {total_params / 1e6:.2f}M parameters")

    criterion = WeedSenseLiverMultiTaskLoss(
        lambda_seg=1.0,
        lambda_cls=0.5,
        lambda_burden=0.3,
        aux_weight=0.4,
        beta_tumor=3.0,
        device=device
    )

    epochs = 15
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/best_weedsense_liver_3way.pth'

    best_score = -1.0
    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_res = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        val_res = evaluate_split(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        # Score balances liver dice, tumor dice, and macro F1
        score = 0.4 * val_res['liver_dice'] + 0.4 * val_res['tumor_dice'] + 0.2 * val_res['macro_f1']
        is_best = score > best_score
        if is_best:
            best_score = score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_res
            }, save_path)

        print(f"Epoch [{epoch:02d}/{epochs}] ({elapsed:.1f}s) | "
              f"Loss: {train_res['total_loss']:.3f} | "
              f"Val Liver Dice: {val_res['liver_dice']:.2f}% | "
              f"Val Tumor Dice: {val_res['tumor_dice']:.2f}% | "
              f"Val Cls F1: {val_res['macro_f1']:.2f}% | "
              f"Val Burden MAE: {val_res['burden_mae']:.2f}% "
              f"{'[*] BEST' if is_best else ''}")

    # Load best checkpoint and evaluate on Test Set
    print("\n--- Evaluating Best WeedSenseLiverNet on Held-out Test Set (30 cases) ---")
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_res = evaluate_split(model, test_loader, criterion, device)

    print("\n" + "=" * 70)
    print("FINAL TEST SET RESULTS: WeedSenseLiverNet on LiMT Dataset")
    print("=" * 70)
    print(f"Segmentation:")
    print(f"  Liver Dice:               {test_res['liver_dice']:.2f}%")
    print(f"  Tumor (Combined) Dice:    {test_res['tumor_dice']:.2f}%")
    for c, name in [(1, 'Liver'), (2, 'Cyst'), (3, 'Hemangioma'), (4, 'Metastasis'), (5, 'HCC')]:
        print(f"    Class {c} ({name:12s}): {test_res['per_class_dice'][c]:.2f}%")
    print(f"\nClassification:")
    print(f"  Hamming Accuracy:         {test_res['hamming_acc']:.2f}%")
    print(f"  Subset Accuracy:          {test_res['subset_acc']:.2f}%")
    print(f"  Macro F1:                 {test_res['macro_f1']:.2f}%")
    print(f"  Micro F1:                 {test_res['micro_f1']:.2f}%")
    print(f"\nTumor Burden Regression:")
    print(f"  MAE:                      {test_res['burden_mae']:.2f}%")
    print(f"  RMSE:                     {test_res['burden_rmse']:.2f}%")
    print(f"  R^2:                      {test_res['burden_r2']:.4f}")
    print("=" * 70)

    # Save summary
    res_df = pd.DataFrame([{
        'Model': 'WeedSenseLiverNet (Ours)',
        'Liver Dice (%)': round(test_res['liver_dice'], 2),
        'Tumor Dice (%)': round(test_res['tumor_dice'], 2),
        'Hamming Acc (%)': round(test_res['hamming_acc'], 2),
        'Macro F1 (%)': round(test_res['macro_f1'], 2),
        'Micro F1 (%)': round(test_res['micro_f1'], 2),
        'Burden MAE (%)': round(test_res['burden_mae'], 2),
        'Burden R2': round(test_res['burden_r2'], 4)
    }])
    os.makedirs('results', exist_ok=True)
    res_df.to_csv('results/weedsense_liver_test_results.csv', index=False)
    print("Results saved to results/weedsense_liver_test_results.csv")


if __name__ == '__main__':
    main()
