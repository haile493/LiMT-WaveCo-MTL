"""
Unified Training Script for WaveCo-MTL and Baseline Architectures on the LiMT Benchmark.
Supports Joint Multi-Task (Segmentation + Classification), Segmentation-Only, and Classification-Only modes.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure local project imports work cleanly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import LiMTMultiTaskDataset
from models.waveco_mtl_net import WaveCo_MTLNet
from models.multitask_net import MultiTaskLiverNet, SingleTaskSegmentationNet, SingleTaskClassificationNet
from models.multitask_msff_net import MultiTaskLiverNet_MSFF
from training.losses import MultiTaskLoss, DiceLoss

CLASS_NAMES = ["Liver (0)", "Cyst (1)", "Hemangioma (2)", "Metastasis (3)", "HCC (4)"]


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Task or Single-Task Models on LiMT Dataset")
    parser.add_argument("--model", type=str, default="waveco_mtl",
                        choices=["waveco_mtl", "multitask_net", "multitask_msff", "seg_only", "cls_only"],
                        help="Network architecture to train")
    parser.add_argument("--data_dir", type=str, default="data/LiMT Dataset",
                        help="Path to LiMT Dataset folder containing ct/ and seg/")
    parser.add_argument("--splits_csv", type=str, default="data/limt_splits.csv",
                        help="Path to stratified splits CSV file")
    parser.add_argument("--split_type", type=str, default="3way", choices=["3way", "5fold"],
                        help="Data split type to use")
    parser.add_argument("--fold", type=int, default=0, help="Fold index when using 5-fold cross validation")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--target_size", type=int, default=64, help="Spatial resolution cube (D=H=W)")
    parser.add_argument("--use_clahe", action="store_true", default=True, help="Apply CLAHE HU window preprocessing")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Output directory for model checkpoints")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    return parser.parse_args()


def build_model(model_name: str, device: torch.device):
    if model_name == "waveco_mtl":
        return WaveCo_MTLNet(in_channels=1, num_seg_classes=2, num_cls_classes=5, base_channels=24).to(device)
    elif model_name == "multitask_net":
        return MultiTaskLiverNet(in_channels=1, num_seg_classes=2, num_cls_classes=5, base_channels=24).to(device)
    elif model_name == "multitask_msff":
        return MultiTaskLiverNet_MSFF(in_channels=1, num_seg_classes=2, num_cls_classes=5, base_channels=24).to(device)
    elif model_name == "seg_only":
        return SingleTaskSegmentationNet(in_channels=1, num_classes=2, base_channels=24).to(device)
    elif model_name == "cls_only":
        return SingleTaskClassificationNet(in_channels=1, num_classes=5, base_channels=24).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    os.makedirs(args.save_dir, exist_ok=True)

    target_shape = (args.target_size, args.target_size, args.target_size)
    print(f"[Config] Model: {args.model} | Epochs: {args.epochs} | Batch: {args.batch_size} | Size: {target_shape}")

    train_ds = LiMTMultiTaskDataset(
        data_dir=args.data_dir,
        splits_csv=args.splits_csv,
        split_type=args.split_type,
        split_val="train" if args.split_type == "3way" else f"train_fold_{args.fold}",
        target_shape=target_shape,
        use_clahe=args.use_clahe,
        augment=True,
        cache_data=True
    )
    val_ds = LiMTMultiTaskDataset(
        data_dir=args.data_dir,
        splits_csv=args.splits_csv,
        split_type=args.split_type,
        split_val="val" if args.split_type == "3way" else f"val_fold_{args.fold}",
        target_shape=target_shape,
        use_clahe=args.use_clahe,
        augment=False,
        cache_data=True
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[Data] Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    model = build_model(args.model, device)
    criterion = MultiTaskLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_score = -1.0
    save_path = os.path.join(args.save_dir, f"best_{args.model}_{args.split_type}.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            seg_targets = batch["segmentation"].to(device)
            cls_targets = batch["classification"].to(device)

            optimizer.zero_grad()
            out = model(imgs)

            if isinstance(out, dict):
                seg_logits = out.get("seg_logits")
                cls_logits = out.get("cls_logits")
            elif args.model == "seg_only":
                seg_logits = out
                cls_logits = None
            elif args.model == "cls_only":
                seg_logits = None
                cls_logits = out

            loss_dict = criterion(seg_logits, cls_logits, seg_targets, cls_targets)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        val_cls_preds, val_cls_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                seg_targets = batch["segmentation"].to(device)
                cls_targets = batch["classification"].to(device)

                out = model(imgs)
                if isinstance(out, dict):
                    seg_logits = out.get("seg_logits")
                    cls_logits = out.get("cls_logits")
                elif args.model == "seg_only":
                    seg_logits = out
                    cls_logits = None
                elif args.model == "cls_only":
                    seg_logits = None
                    cls_logits = out

                loss_dict = criterion(seg_logits, cls_logits, seg_targets, cls_targets)
                val_loss += loss_dict["loss"].item() * imgs.size(0)

                if cls_logits is not None:
                    probs = torch.sigmoid(cls_logits).cpu().numpy()
                    val_cls_preds.append((probs >= 0.5).astype(int))
                    val_cls_targets.append(cls_targets.cpu().numpy().astype(int))

        val_loss /= len(val_ds)
        val_f1 = 0.0
        if len(val_cls_preds) > 0:
            preds_all = np.vstack(val_cls_preds)
            targets_all = np.vstack(val_cls_targets)
            val_f1 = f1_score(targets_all, preds_all, average="macro", zero_division=0) * 100.0

        current_score = -val_loss if val_f1 == 0.0 else val_f1
        is_best = current_score > best_score
        if is_best:
            best_score = current_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_f1": val_f1,
                "args": vars(args)
            }, save_path)

        dt = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Macro-F1: {val_f1:.2f}% {'[BEST]' if is_best else ''}")

    print(f"\n[Done] Training completed! Best checkpoint preserved at: {save_path}")


if __name__ == "__main__":
    main()
