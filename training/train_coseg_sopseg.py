"""
Training and Evaluation script for combined Co-Seg + SOPSeg Architecture on LiMT CT Dataset.
Mutual Prompt Collaborative Learning + Multi-Scale Edge-Aware Progressive Refinement.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from dataset import LiMTMultiTaskDataset
from models.sopseg_liver import CoSeg_SOPSeg_LiverNet
from losses_coseg_sopseg import CoSegSOPSegLoss
from metrics import evaluate_segmentation_batch


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss_accum = 0.0
    prm_loss_accum = 0.0
    prior_loss_accum = 0.0

    for batch in dataloader:
        images = batch['image'].to(device)
        seg_targets = batch['segmentation'].to(device)
        cls_targets = batch['classification'].to(device)
        burden_targets = batch['tumor_burden'].to(device)

        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, seg_targets, cls_targets, burden_targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        total_loss_accum += loss.item()
        prm_loss_accum += loss_dict.get('loss_prm', 0.0)
        prior_loss_accum += loss_dict.get('loss_prior', 0.0)

    n = len(dataloader)
    return {
        'total_loss': total_loss_accum / n,
        'prm_loss': prm_loss_accum / n,
        'prior_loss': prior_loss_accum / n
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

        # Segmentation metrics on refined outputs
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
        'liver_dice': float(np.mean(seg_dices_liver)),
        'tumor_dice': float(np.mean(seg_dices_tumor)),
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
    print(f"Executing Co-Seg + SOPSeg training on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_ds = LiMTMultiTaskDataset(
        data_dir="LiMT Dataset",
        splits_csv="LiMT Dataset/limt_splits.csv",
        split_type="3way",
        split_val="train",
        target_shape=(64, 64, 64),
        use_clahe=True,
        augment=True,
        cache_data=True
    )
    val_ds = LiMTMultiTaskDataset(
        data_dir="LiMT Dataset",
        splits_csv="LiMT Dataset/limt_splits.csv",
        split_type="3way",
        split_val="val",
        target_shape=(64, 64, 64),
        use_clahe=True,
        augment=False,
        cache_data=True
    )
    test_ds = LiMTMultiTaskDataset(
        data_dir="LiMT Dataset",
        splits_csv="LiMT Dataset/limt_splits.csv",
        split_type="3way",
        split_val="test",
        target_shape=(64, 64, 64),
        use_clahe=True,
        augment=False,
        cache_data=True
    )

    batch_size = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("Building CoSeg_SOPSeg_LiverNet...")
    model = CoSeg_SOPSeg_LiverNet(
        in_channels=1,
        num_seg_classes=6,
        num_cls_classes=5,
        base_channels=24
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Co-Seg + SOPSeg initialized with {total_params / 1e6:.2f}M trainable parameters.")

    criterion = CoSegSOPSegLoss(
        lambda_prior=0.5,
        lambda_tumor=2.5,
        lambda_edge=1.0,
        lambda_cls=0.5,
        lambda_burden=0.1
    )

    epochs = 15
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/best_coseg_sopseg_3way.pth'

    best_score = -1.0
    print("\n--- Starting Co-Seg + SOPSeg Combined Training ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_res = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        scheduler.step()

        val_res = evaluate_split(model, val_loader, criterion, device)
        elapsed = time.time() - t0

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
              f"Loss: {train_res['total_loss']:.3f} (PRM: {train_res['prm_loss']:.2f}, Prior: {train_res['prior_loss']:.2f}) | "
              f"Val Liver Dice: {val_res['liver_dice']:.2f}% | "
              f"Val Tumor Dice: {val_res['tumor_dice']:.2f}% | "
              f"Val Cls F1: {val_res['macro_f1']:.2f}% | "
              f"Val Burden MAE: {val_res['burden_mae']:.2f}% "
              f"{'[*] BEST' if is_best else ''}")

    # Load best checkpoint and evaluate on Test Set
    print("\n--- Evaluating Best Co-Seg + SOPSeg on Held-out Test Set (30 cases) ---")
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_res = evaluate_split(model, test_loader, criterion, device)

    print("\n" + "=" * 70)
    print("FINAL TEST SET RESULTS: Co-Seg + SOPSeg on LiMT Dataset")
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
        'Model': 'Co-Seg + SOPSeg (Combined)',
        'Liver Dice (%)': round(test_res['liver_dice'], 2),
        'Tumor Dice (%)': round(test_res['tumor_dice'], 2),
        'Hamming Acc (%)': round(test_res['hamming_acc'], 2),
        'Macro F1 (%)': round(test_res['macro_f1'], 2),
        'Micro F1 (%)': round(test_res['micro_f1'], 2),
        'Burden MAE (%)': round(test_res['burden_mae'], 2),
        'Burden R2': round(test_res['burden_r2'], 4)
    }])
    os.makedirs('results', exist_ok=True)
    res_df.to_csv('results/coseg_sopseg_test_results.csv', index=False)
    print("Results saved to results/coseg_sopseg_test_results.csv")


if __name__ == '__main__':
    main()
