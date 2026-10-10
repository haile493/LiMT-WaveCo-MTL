"""
Training and Evaluation script for WaveAttn-SOPSeg and Co-Seg + WaveAttn-SOPSeg on LiMT CT Dataset.
Integrates:
- 3D Haar Discrete Wavelet Transform (HaarDWT3D / HaarIDWT3D)
- 3D WaveFusion multi-scale progressive refinement
- GLCAM-3D context attention gating
- Co-Seg mutual prompt collaborative learning
"""

import os
import sys
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
from models.waveattn_sopseg_liver import WaveAttn_SOPSegLiverNet, CoSeg_WaveAttn_SOPSegLiverNet
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

        # Classification
        cls_probs = torch.sigmoid(outputs['cls_logits']).cpu().numpy()
        cls_preds_all.append(cls_probs)
        cls_targets_all.append(cls_targets.cpu().numpy())

        # Tumor Burden
        burden_preds_all.append(outputs['tumor_burden'].cpu().numpy())
        burden_targets_all.append(burden_targets.cpu().numpy())

    n = len(dataloader)
    cls_preds_all = np.concatenate(cls_preds_all, axis=0)
    cls_targets_all = np.concatenate(cls_targets_all, axis=0)
    burden_preds_all = np.concatenate(burden_preds_all, axis=0).flatten()
    burden_targets_all = np.concatenate(burden_targets_all, axis=0).flatten()

    # Classification Metrics (binary threshold 0.5)
    cls_preds_bin = (cls_preds_all >= 0.5).astype(int)
    cls_f1_macro = f1_score(cls_targets_all, cls_preds_bin, average='macro', zero_division=0)
    cls_acc_subset = accuracy_score(cls_targets_all, cls_preds_bin)

    # Burden Metrics
    burden_mae = mean_absolute_error(burden_targets_all, burden_preds_all)
    burden_r2 = r2_score(burden_targets_all, burden_preds_all)

    mean_liver_dice = float(np.mean(seg_dices_liver))
    mean_tumor_dice = float(np.mean(seg_dices_tumor))

    class_names = {1: 'Liver', 2: 'Cyst', 3: 'Hemangioma', 4: 'Metastasis', 5: 'HCC'}
    per_class_summary = {class_names[c]: (float(np.mean(per_class_dices[c])) if len(per_class_dices[c]) > 0 else 0.0) for c in range(1, 6)}

    return {
        'eval_loss': total_loss_accum / n,
        'liver_dice': mean_liver_dice,
        'tumor_dice': mean_tumor_dice,
        'per_class_dice': per_class_summary,
        'cls_f1_macro': cls_f1_macro,
        'cls_subset_acc': cls_acc_subset,
        'burden_mae': burden_mae,
        'burden_r2': burden_r2
    }


def main():
    parser = argparse.ArgumentParser(description="Train WaveAttn-SOPSeg on LiMT Dataset")
    parser.add_argument('--model_type', type=str, default='combined', choices=['combined', 'standalone'],
                        help="Model architecture: 'combined' (CoSeg + WaveAttn SOPSeg) or 'standalone' (WaveAttn SOPSeg)")
    parser.add_argument('--epochs', type=int, default=30, help="Training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--seed', type=int, default=1337, help="Random seed")
    parser.add_argument('--output_dir', type=str, default='checkpoints', help="Checkpoint output directory")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 1. Dataset & DataLoader (3-way split: 60 train, 10 val, 30 test)
    train_dataset = LiMTMultiTaskDataset(
        data_dir="LiMT Dataset",
        splits_csv="LiMT Dataset/limt_splits.csv",
        split_type="3way",
        split_val="train",
        target_shape=(64, 64, 64),
        use_clahe=True,
        augment=True,
        cache_data=True
    )
    val_dataset = LiMTMultiTaskDataset(
        data_dir="LiMT Dataset",
        splits_csv="LiMT Dataset/limt_splits.csv",
        split_type="3way",
        split_val="val",
        target_shape=(64, 64, 64),
        use_clahe=True,
        augment=False,
        cache_data=True
    )
    test_dataset = LiMTMultiTaskDataset(
        data_dir="LiMT Dataset",
        splits_csv="LiMT Dataset/limt_splits.csv",
        split_type="3way",
        split_val="test",
        target_shape=(64, 64, 64),
        use_clahe=True,
        augment=False,
        cache_data=True
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Splits loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # 2. Model Selection
    if args.model_type == 'combined':
        print("\n>>> Building Co-Seg + WaveAttn-SOPSeg Combined Model...")
        model = CoSeg_WaveAttn_SOPSegLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=24).to(device)
        ckpt_name = "best_coseg_waveattn_sopseg_3way.pth"
        results_csv = "results/coseg_waveattn_sopseg_test_results.csv"
    else:
        print("\n>>> Building Standalone WaveAttn-SOPSeg Model...")
        model = WaveAttn_SOPSegLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=24).to(device)
        ckpt_name = "best_waveattn_sopseg_3way.pth"
        results_csv = "results/waveattn_sopseg_test_results.csv"

    ckpt_path = os.path.join(args.output_dir, ckpt_name)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type} | Trainable Parameters: {total_params:,}")

    # 3. Loss & Optimizer
    criterion = CoSegSOPSegLoss(
        lambda_prior=0.5,
        lambda_tumor=2.5,
        lambda_edge=1.0,
        lambda_cls=0.5,
        lambda_burden=0.1,
        bce_cls_pos_weight=2.0
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # 4. Training Loop
    best_val_tumor_dice = -1.0
    best_val_score = -1.0
    history = []

    print(f"\nStarting {args.model_type.upper()} Training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_res = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_res = evaluate_split(model, val_loader, criterion, device)
        scheduler.step()
        ep_duration = time.time() - ep_start

        # Composite score prioritizing tumor dice with liver stabilization
        composite_score = 0.6 * val_res['tumor_dice'] + 0.3 * val_res['liver_dice'] + 0.1 * val_res['cls_f1_macro']

        history.append({
            'epoch': epoch,
            'train_loss': train_res['total_loss'],
            'val_loss': val_res['eval_loss'],
            'val_liver_dice': val_res['liver_dice'],
            'val_tumor_dice': val_res['tumor_dice'],
            'val_cls_f1': val_res['cls_f1_macro'],
            'val_burden_mae': val_res['burden_mae'],
            'lr': scheduler.get_last_lr()[0]
        })

        is_best = composite_score > best_val_score
        if is_best:
            best_val_score = composite_score
            best_val_tumor_dice = val_res['tumor_dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_score': best_val_score,
                'val_metrics': val_res,
                'model_type': args.model_type
            }, ckpt_path)
            mark = "(*BEST*)"
        else:
            mark = ""

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_duration:.1f}s) | "
            f"Train Loss: {train_res['total_loss']:.4f} | "
            f"Val Loss: {val_res['eval_loss']:.4f} | "
            f"Liver Dice: {val_res['liver_dice']:.2f}% | "
            f"Tumor Dice: {val_res['tumor_dice']:.2f}% | "
            f"Cls F1: {val_res['cls_f1_macro']*100:.2f}% {mark}"
        )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best Val Score: {best_val_score:.4f} (Tumor Dice: {best_val_tumor_dice:.2f}%)")

    # 5. Load Best Model & Run Comprehensive Test Evaluation
    print(f"\n========================================================")
    print(f"  EVALUATING BEST CHECKPOINT ON HELD-OUT TEST SPLIT (N=30)  ")
    print(f"========================================================")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_res = evaluate_split(model, test_loader, criterion, device)

    print(f"\n--- Test Results for {args.model_type.upper()} ---")
    print(f"  Liver Dice (%)         : {test_res['liver_dice']:.2f}%")
    print(f"  Overall Tumor Dice (%) : {test_res['tumor_dice']:.2f}%")
    print(f"  Multi-label Macro F1   : {test_res['cls_f1_macro']*100:.2f}%")
    print(f"  Subset Accuracy (%)    : {test_res['cls_subset_acc']*100:.2f}%")
    print(f"  Tumor Burden MAE       : {test_res['burden_mae']:.4f}")
    print(f"  Tumor Burden R2        : {test_res['burden_r2']:.4f}")

    print("\n--- Per-Class Lesion Dice Breakdown ---")
    for k, v in test_res['per_class_dice'].items():
        print(f"  {k:12s}: {v*100:.2f}%")

    # Save detailed evaluation CSV
    test_df = pd.DataFrame([{
        'Model': f"WaveAttn-SOPSeg ({args.model_type})",
        'Liver Dice (%)': test_res['liver_dice'],
        'Tumor Dice (%)': test_res['tumor_dice'],
        'Cyst Dice (%)': test_res['per_class_dice']['Cyst'] * 100,
        'Hemangioma Dice (%)': test_res['per_class_dice']['Hemangioma'] * 100,
        'Metastasis Dice (%)': test_res['per_class_dice']['Metastasis'] * 100,
        'HCC Dice (%)': test_res['per_class_dice']['HCC'] * 100,
        'Cls Macro F1 (%)': test_res['cls_f1_macro'] * 100,
        'Burden MAE': test_res['burden_mae'],
        'Burden R2': test_res['burden_r2']
    }])
    test_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")


if __name__ == '__main__':
    main()
