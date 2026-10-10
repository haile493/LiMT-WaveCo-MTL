import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import LiMTMultiTaskDataset
from models.resvnet_fusion import ResVNetFusion
from losses_ref78 import Ref78MultiTaskLoss
from metrics import evaluate_segmentation_batch, evaluate_classification_batch

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Ref [78] ResVNetFusion Multi-Task Pipeline")
    parser.add_argument("--data_dir", type=str, default="LiMT Dataset")
    parser.add_argument("--splits_csv", type=str, default="LiMT Dataset/limt_splits.csv")
    parser.add_argument("--split_type", type=str, default="3way", choices=["3way", "kfold"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--target_size", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for segmentation loss in Ref [78]")
    parser.add_argument("--tumor_weight", type=float, default=3.0, help="Weight for tumor vs liver in seg loss")
    parser.add_argument("--hu_min", type=float, default=-200.0, help="HU min from Ref [78]")
    parser.add_argument("--hu_max", type=float, default=340.0, help="HU max from Ref [78]")
    parser.add_argument("--use_clahe", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--cache_data", action="store_true", default=True)
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss_accum = 0.0
    seg_loss_accum = 0.0
    cls_loss_accum = 0.0
    tumor_dice_accum = 0.0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        seg_targets = batch['segmentation'].to(device)
        cls_targets = batch['classification'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss, loss_dict = criterion(outputs, seg_targets, cls_targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss_accum += loss.item()
        seg_loss_accum += loss_dict.get('loss_seg', 0)
        cls_loss_accum += loss_dict.get('loss_cls', 0)
        tumor_dice_accum += loss_dict.get('loss_tumor_dice', 0)
        
    n = len(dataloader)
    return {
        'total_loss': total_loss_accum / n,
        'seg_loss': seg_loss_accum / n,
        'cls_loss': cls_loss_accum / n,
        'tumor_loss': tumor_dice_accum / n
    }

def validate(model, dataloader, criterion, device, tumor_thresh=0.5):
    model.eval()
    total_loss_accum = 0.0
    seg_dices_liver = []
    seg_dices_tumor = []
    cls_preds_all = []
    cls_targets_all = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            seg_targets = batch['segmentation'].to(device)
            cls_targets = batch['classification'].to(device)
            
            outputs = model(images)
            loss, _ = criterion(outputs, seg_targets, cls_targets)
            total_loss_accum += loss.item()
            
            seg_res = evaluate_segmentation_batch(outputs['seg_logits'], seg_targets, tumor_thresh=tumor_thresh)
            seg_dices_liver.append(seg_res['liver_dice'])
            seg_dices_tumor.append(seg_res['tumor_dice'])
            
            cls_preds_all.append(outputs['cls_logits'].cpu())
            cls_targets_all.append(batch['classification'].cpu())
            
    n = len(dataloader)
    all_logits = torch.cat(cls_preds_all, dim=0)
    all_targets = torch.cat(cls_targets_all, dim=0)
    cls_res = evaluate_classification_batch(all_logits, all_targets)
    
    return {
        'val_loss': total_loss_accum / n,
        'liver_dice': np.mean(seg_dices_liver),
        'tumor_dice': np.mean(seg_dices_tumor),
        'accuracy': cls_res['hamming_acc'],
        'subset_acc': cls_res['subset_acc'],
        'precision': cls_res['precision'],
        'recall': cls_res['recall'],
        'f1': cls_res['f1'],
        'per_class_f1': cls_res['per_class_f1']
    }

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    os.makedirs(args.save_dir, exist_ok=True)
    
    target_shape = (args.target_size, args.target_size, args.target_size)
    
    train_ds = LiMTMultiTaskDataset(
        data_dir=args.data_dir, splits_csv=args.splits_csv,
        split_type=args.split_type, split_val="train" if args.split_type == "3way" else ("train", args.fold),
        target_shape=target_shape, hu_min=args.hu_min, hu_max=args.hu_max,
        use_clahe=args.use_clahe, augment=True, cache_data=args.cache_data
    )
    val_ds = LiMTMultiTaskDataset(
        data_dir=args.data_dir, splits_csv=args.splits_csv,
        split_type=args.split_type, split_val="val" if args.split_type == "3way" else ("val", args.fold),
        target_shape=target_shape, hu_min=args.hu_min, hu_max=args.hu_max,
        use_clahe=args.use_clahe, augment=False, cache_data=args.cache_data
    )
    test_ds = LiMTMultiTaskDataset(
        data_dir=args.data_dir, splits_csv=args.splits_csv,
        split_type=args.split_type, split_val="test" if args.split_type == "3way" else ("val", args.fold),
        target_shape=target_shape, hu_min=args.hu_min, hu_max=args.hu_max,
        use_clahe=args.use_clahe, augment=False, cache_data=args.cache_data
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize ResVNetFusion
    model = ResVNetFusion(in_channels=1, num_seg_classes=2, num_cls_classes=5, base_channels=16).to(device)
    criterion = Ref78MultiTaskLoss(alpha=args.alpha, tumor_weight=args.tumor_weight).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_score = -float('inf')
    best_ckpt_path = os.path.join(args.save_dir, f"best_resvnet_fusion_{args.split_type}.pth")
    
    print(f"\n--- Starting Training (Ref [78] ResVNetFusion) for {args.epochs} Epochs ---")
    for epoch in range(1, args.epochs + 1):
        train_res = train_epoch(model, train_loader, optimizer, criterion, device)
        val_res = validate(model, val_loader, criterion, device, tumor_thresh=0.5)
        scheduler.step()
        
        # Primary score combines Tumor Dice and Liver Dice and Cls F1
        score = val_res['tumor_dice'] * 3.0 + val_res['liver_dice'] + val_res['f1']
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {train_res['total_loss']:.3f} | "
              f"Val Loss: {val_res['val_loss']:.3f} | Liver Dice: {val_res['liver_dice']:.1f}% | "
              f"Tumor Dice: {val_res['tumor_dice']:.1f}% | Cls Acc: {val_res['accuracy']:.1f}% | Cls F1: {val_res['f1']:.1f}%")
              
        if score > best_score:
            best_score = score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_res,
                'args': vars(args)
            }, best_ckpt_path)
            
    print(f"\nBest model saved to {best_ckpt_path}")
    
    # Run test evaluation with calibrated thresholds
    print("\n--- Running Final Evaluation on Test Split ---")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    
    for th in [0.3, 0.4, 0.5, 0.6]:
        test_m = validate(model, test_loader, criterion, device, tumor_thresh=th)
        print(f"Threshold {th:.2f} -> Liver Dice: {test_m['liver_dice']:.2f}% | "
              f"Tumor Dice: {test_m['tumor_dice']:.2f}% | Cls Acc: {test_m['accuracy']:.2f}% | Cls F1: {test_m['f1']:.2f}%")

if __name__ == '__main__':
    main()
