"""
Comprehensive evaluation of WeedSenseLiverNet on LiMT held-out test set (30 cases).
Evaluates:
1. Segmentation: Liver Dice, Tumor Dice, Individual Lesion Dices (Cyst, Hemangioma, Met, HCC)
2. Classification: Hamming Acc, Subset Acc, Macro Prec, Macro Rec, Macro F1, Micro F1, Macro AUC-ROC, Per-Class Scores
3. Quantitative Regression: Tumor Burden MAE, RMSE, R^2
4. Model Complexity & Inference Speed: Params, GFLOPs, Latency (ms), FPS
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)
from thop import profile

from dataset import LiMTMultiTaskDataset
from models.weedsense_liver import WeedSenseLiverNet
from metrics import evaluate_segmentation_batch

CLASS_NAMES = ["Liver (0)", "Cyst (1)", "Hemangioma (2)", "Metastasis (3)", "HCC (4)"]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = 'checkpoints/best_weedsense_liver_3way.pth'

    print(f"Loading WeedSenseLiverNet from {ckpt_path} on {device}...")
    model = WeedSenseLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Complexity & FLOPs
    x_dummy = torch.randn(1, 1, 64, 64, 64, device=device)
    flops, _ = profile(model, inputs=(x_dummy,), verbose=False)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = flops / 1e9

    # Benchmark Speed
    torch.cuda.synchronize()
    for _ in range(20):
        _ = model(x_dummy)
    torch.cuda.synchronize()

    timings = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x_dummy)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
    latency_ms = np.mean(timings) * 1000.0
    fps = 1.0 / np.mean(timings)

    # Dataset
    test_ds = LiMTMultiTaskDataset(split_type="3way", split_val="test", target_shape=(64, 64, 64), use_clahe=True, augment=False, cache_data=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # Evaluate
    all_seg_liver_dices = []
    all_seg_tumor_dices = []
    all_cls_logits = []
    all_cls_targets = []
    all_burden_preds = []
    all_burden_targets = []

    per_class_seg_dice = {c: [] for c in range(1, 6)}

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            seg_targets = batch['segmentation'].to(device)
            cls_targets = batch['classification']
            burden_targets = batch['tumor_burden']

            out = model(imgs)

            seg_logits = out['seg_logits']
            seg_preds = torch.argmax(seg_logits, dim=1)

            # Dice calculation using standardized metrics.py
            seg_m = evaluate_segmentation_batch(seg_logits, seg_targets)
            all_seg_liver_dices.append(seg_m['liver_dice'])
            all_seg_tumor_dices.append(seg_m['tumor_dice'])

            # Per class segmentation dice
            for c in range(1, 6):
                p_c = (seg_preds == c).float().cpu().numpy()
                t_c = (seg_targets == c).float().cpu().numpy()
                for b in range(imgs.shape[0]):
                    card = p_c[b].sum() + t_c[b].sum()
                    if card > 0:
                        inter = np.logical_and(p_c[b], t_c[b]).sum()
                        dice_val = (2.0 * inter + 1e-5) / (card + 1e-5) * 100.0
                        per_class_seg_dice[c].append(dice_val)

            all_cls_logits.append(out['cls_logits'].cpu())
            all_cls_targets.append(cls_targets)

            all_burden_preds.extend(out['tumor_burden'].cpu().numpy().flatten().tolist())
            all_burden_targets.extend(burden_targets.numpy().flatten().tolist())

    # Classification Metrics
    logits_arr = torch.cat(all_cls_logits, dim=0).numpy()
    targets_arr = torch.cat(all_cls_targets, dim=0).numpy().astype(int)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))
    preds_bin = (probs_arr >= 0.5).astype(int)

    hamming_acc = float((targets_arr == preds_bin).mean() * 100.0)
    subset_acc = float(accuracy_score(targets_arr, preds_bin) * 100.0)
    macro_prec = float(precision_score(targets_arr, preds_bin, average='macro', zero_division=0) * 100.0)
    macro_rec = float(recall_score(targets_arr, preds_bin, average='macro', zero_division=0) * 100.0)
    macro_f1 = float(f1_score(targets_arr, preds_bin, average='macro', zero_division=0) * 100.0)
    micro_f1 = float(f1_score(targets_arr, preds_bin, average='micro', zero_division=0) * 100.0)

    auc_scores = []
    for c in range(5):
        if len(np.unique(targets_arr[:, c])) > 1:
            try:
                auc_c = roc_auc_score(targets_arr[:, c], probs_arr[:, c]) * 100.0
                auc_scores.append(auc_c)
            except Exception:
                pass
    macro_auc = float(np.mean(auc_scores)) if auc_scores else 0.0

    # Burden Regression Metrics
    bp = np.array(all_burden_preds)
    bt = np.array(all_burden_targets)
    b_mae = float(mean_absolute_error(bt, bp))
    b_rmse = float(np.sqrt(mean_squared_error(bt, bp)))
    b_r2 = float(r2_score(bt, bp)) if len(np.unique(bt)) > 1 else 0.0

    liver_dice = float(np.mean(all_seg_liver_dices))
    tumor_dice = float(np.mean(all_seg_tumor_dices))

    print("\n" + "=" * 80)
    print("STANDARDIZED EVALUATION: WeedSenseLiverNet on LiMT Test Set")
    print("=" * 80)
    print(f"Computational Efficiency:")
    print(f"  Parameters:        {total_params:.2f} M")
    print(f"  GFLOPs:            {gflops:.2f}")
    print(f"  Latency:           {latency_ms:.2f} ms")
    print(f"  Inference Speed:   {fps:.1f} FPS")
    print(f"\n1. Multi-Class Segmentation:")
    print(f"  Liver Dice:        {liver_dice:.2f}%")
    print(f"  Tumor Dice:        {tumor_dice:.2f}%")
    for c, name in [(1, 'Liver'), (2, 'Cyst'), (3, 'Hemangioma'), (4, 'Metastasis'), (5, 'HCC')]:
        d_val = float(np.mean(per_class_seg_dice[c])) if len(per_class_seg_dice[c]) > 0 else 0.0
        print(f"    Class {c} ({name:12s}): {d_val:.2f}%")

    print(f"\n2. Multi-Label Lesion Classification:")
    print(f"  Hamming Accuracy:  {hamming_acc:.2f}%")
    print(f"  Subset Accuracy:   {subset_acc:.2f}%")
    print(f"  Macro Precision:   {macro_prec:.2f}%")
    print(f"  Macro Recall:      {macro_rec:.2f}%")
    print(f"  Macro F1:          {macro_f1:.2f}%")
    print(f"  Micro F1:          {micro_f1:.2f}%")
    print(f"  Macro AUC-ROC:     {macro_auc:.2f}%")

    print(f"\n3. Quantitative Tumor Burden Regression:")
    print(f"  MAE:               {b_mae:.2f}%")
    print(f"  RMSE:              {b_rmse:.2f}%")
    print(f"  R^2:               {b_r2:.4f}")
    print("=" * 80)

    # Save to CSV
    os.makedirs('results', exist_ok=True)
    summary_df = pd.DataFrame([{
        "Model": "WeedSenseLiverNet (3D UIB + BGA + Transformer TGD)",
        "Params (M)": round(total_params, 2),
        "GFLOPs": round(gflops, 2),
        "FPS": round(fps, 1),
        "Liver Dice (%)": round(liver_dice, 2),
        "Tumor Dice (%)": round(tumor_dice, 2),
        "Hamming Acc (%)": round(hamming_acc, 2),
        "Subset Acc (%)": round(subset_acc, 2),
        "Macro F1 (%)": round(macro_f1, 2),
        "Micro F1 (%)": round(micro_f1, 2),
        "Macro AUC-ROC (%)": round(macro_auc, 2),
        "Burden MAE (%)": round(b_mae, 2),
        "Burden R2": round(b_r2, 4)
    }])
    summary_df.to_csv('results/weedsense_liver_evaluation_summary.csv', index=False)
    print("Saved summary to results/weedsense_liver_evaluation_summary.csv")


if __name__ == '__main__':
    main()
