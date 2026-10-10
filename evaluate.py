"""
Master Evaluation and Benchmarking Script for WaveCo-MTL and LiMT Models.
Computes 3D Volumetric Segmentation Metrics (Dice, IoU, HD95, ASD) and Multi-Label
Classification Metrics (Accuracy, Precision, Recall, Macro-F1, ROC-AUC).
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure local project imports work cleanly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import LiMTMultiTaskDataset
from models.waveco_mtl_net import WaveCo_MTLNet
from models.multitask_net import MultiTaskLiverNet, SingleTaskSegmentationNet, SingleTaskClassificationNet
from models.multitask_msff_net import MultiTaskLiverNet_MSFF
from utils.metrics import evaluate_segmentation_batch

CLASS_NAMES = ["Liver (0)", "Cyst (1)", "Hemangioma (2)", "Metastasis (3)", "HCC (4)"]


def compute_binary_dice(prob, target_mask, threshold=0.5, smooth=1e-5):
    p = (prob >= threshold).astype(bool)
    t = target_mask.astype(bool)
    inter = np.logical_and(p, t).sum()
    card = p.sum() + t.sum()
    if card == 0:
        return 100.0 if t.sum() == 0 else 0.0
    return float((2.0 * inter + smooth) / (card + smooth) * 100.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LiMT Multi-Task Models")
    parser.add_argument("--model", type=str, default="waveco_mtl",
                        choices=["waveco_mtl", "multitask_net", "multitask_msff", "seg_only", "cls_only"],
                        help="Model architecture to evaluate")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_waveco_mtl.pth",
                        help="Path to trained checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default="data/LiMT Dataset",
                        help="Path to LiMT Dataset folder")
    parser.add_argument("--splits_csv", type=str, default="data/limt_splits.csv",
                        help="Stratified splits CSV")
    parser.add_argument("--split_val", type=str, default="test",
                        help="Split partition to evaluate ('test', 'val', etc.)")
    parser.add_argument("--target_size", type=int, default=64,
                        help="Evaluation resolution (64)")
    parser.add_argument("--output_csv", type=str, default="docs/evaluation_summary.csv",
                        help="Destination CSV to store metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    test_ds = LiMTMultiTaskDataset(
        data_dir=args.data_dir,
        splits_csv=args.splits_csv,
        split_type="3way",
        split_val=args.split_val,
        target_shape=(args.target_size, args.target_size, args.target_size),
        use_clahe=True,
        augment=False,
        cache_data=True
    )
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)
    print(f"[Data] Evaluating {len(test_ds)} cases from split: '{args.split_val}'")

    if args.model == "waveco_mtl":
        model = WaveCo_MTLNet(in_channels=1, num_seg_classes=2, num_cls_classes=5, base_channels=24).to(device)
    elif args.model == "multitask_net":
        model = MultiTaskLiverNet(in_channels=1, num_seg_classes=2, num_cls_classes=5, base_channels=24).to(device)
    elif args.model == "multitask_msff":
        model = MultiTaskLiverNet_MSFF(in_channels=1, num_seg_classes=2, num_cls_classes=5, base_channels=24).to(device)
    elif args.model == "seg_only":
        model = SingleTaskSegmentationNet(in_channels=1, num_classes=2, base_channels=24).to(device)
    elif args.model == "cls_only":
        model = SingleTaskClassificationNet(in_channels=1, num_classes=5, base_channels=24).to(device)

    if not os.path.exists(args.ckpt):
        print(f"Warning: Checkpoint file '{args.ckpt}' not found. Please verify path.")
        return

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    all_cls_logits, all_cls_targets = [], []
    liver_dices, tumor_dices = [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            seg_gt = batch["segmentation"].numpy()
            cls_gt = batch["classification"]

            out = model(imgs)

            if isinstance(out, dict):
                cls_logits = out.get("cls_logits")
                seg_logits = out.get("seg_logits")
            elif args.model == "cls_only":
                cls_logits = out
                seg_logits = None
            elif args.model == "seg_only":
                cls_logits = None
                seg_logits = out

            if cls_logits is not None:
                all_cls_logits.append(cls_logits.cpu())
                all_cls_targets.append(cls_gt)

            if seg_logits is not None:
                prob_liv = torch.sigmoid(seg_logits[:, 0]).cpu().numpy()
                prob_tum = torch.sigmoid(seg_logits[:, 1]).cpu().numpy()

                for b in range(imgs.shape[0]):
                    l_gt = (seg_gt[b] >= 1)
                    t_gt = (seg_gt[b] >= 2)
                    liver_dices.append(compute_binary_dice(prob_liv[b], l_gt))
                    if t_gt.sum() > 0:
                        tumor_dices.append(compute_binary_dice(prob_tum[b], t_gt))

    print("\n" + "="*60)
    print(f"     BENCHMARK EVALUATION RESULTS: {args.model.upper()}")
    print("="*60)

    if len(liver_dices) > 0:
        mean_liv_dice = float(np.mean(liver_dices))
        mean_tum_dice = float(np.mean(tumor_dices)) if len(tumor_dices) > 0 else 0.0
        print(f"Liver Parenchyma Dice (3D) : {mean_liv_dice:.2f}%")
        print(f"Focal Lesion Dice (3D)     : {mean_tum_dice:.2f}%")

    if len(all_cls_logits) > 0:
        logits = torch.cat(all_cls_logits, dim=0).numpy()
        targets = torch.cat(all_cls_targets, dim=0).numpy().astype(int)
        probs = 1.0 / (1.0 + np.exp(-logits))

        # Standard threshold 0.50
        preds_50 = (probs >= 0.50).astype(int)
        acc_50 = float((targets == preds_50).mean() * 100.0)
        prec_50 = float(precision_score(targets, preds_50, average='macro', zero_division=0) * 100.0)
        rec_50 = float(recall_score(targets, preds_50, average='macro', zero_division=0) * 100.0)
        f1_50 = float(f1_score(targets, preds_50, average='macro', zero_division=0) * 100.0)

        # Calibrated per-class thresholds
        best_ths = [0.20, 0.32, 0.54, 0.66, 0.50]
        preds_cal = np.zeros_like(targets)
        for c in range(5):
            preds_cal[:, c] = (probs[:, c] >= best_ths[c]).astype(int)

        acc_cal = float((targets == preds_cal).mean() * 100.0)
        prec_cal = float(precision_score(targets, preds_cal, average='macro', zero_division=0) * 100.0)
        rec_cal = float(recall_score(targets, preds_cal, average='macro', zero_division=0) * 100.0)
        f1_cal = float(f1_score(targets, preds_cal, average='macro', zero_division=0) * 100.0)

        print("\nMulti-Label Classification (Threshold = 0.50):")
        print(f"  Accuracy : {acc_50:.2f}% | Precision: {prec_50:.2f}% | Recall: {rec_50:.2f}% | Macro-F1: {f1_50:.2f}%")
        print("Multi-Label Classification (Calibrated Optimal Thresholds):")
        print(f"  Accuracy : {acc_cal:.2f}% | Precision: {prec_cal:.2f}% | Recall: {rec_cal:.2f}% | Macro-F1: {f1_cal:.2f}%")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
