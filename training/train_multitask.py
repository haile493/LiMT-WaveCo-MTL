import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import LiMTMultiTaskDataset
from models.multitask_net import MultiTaskLiverNet, SingleTaskSegmentationNet, SingleTaskClassificationNet
from models.multitask_msff_net import MultiTaskLiverNet_MSFF
from losses import TumorAwareMultiTaskLoss
from metrics import evaluate_segmentation_batch, evaluate_classification_batch

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="LiMT Multi-Task Benchmark Pipeline with MSFF")
    parser.add_argument("--data_dir", type=str, default="LiMT Dataset", help="Path to LiMT Dataset folder")
    parser.add_argument("--splits_csv", type=str, default="LiMT Dataset/limt_splits.csv", help="Path to splits csv")
    parser.add_argument("--mode", type=str, default="multitask", choices=["multitask", "seg_only", "cls_only"], help="Model mode")
    parser.add_argument("--model_arch", type=str, default="msff", choices=["msff", "baseline"], help="Model architecture")
    parser.add_argument("--split_type", type=str, default="3way", choices=["3way", "kfold"], help="Dataset split scheme")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for kfold cross-validation")
    parser.add_argument("--target_size", type=int, default=64, help="Target spatial size (D=H=W)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lambda_seg", type=float, default=1.0, help="Segmentation loss weight")
    parser.add_argument("--lambda_cls", type=float, default=0.5, help="Classification loss weight")
    parser.add_argument("--beta_tumor", type=float, default=3.0, help="Tumor-specific Dice loss weight")
    parser.add_argument("--use_clahe", action="store_true", default=True, help="Apply CLAHE enhancement")
    parser.add_argument("--no_clahe", dest="use_clahe", action="store_false", help="Disable CLAHE enhancement")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation on test set")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint for evaluation")
    parser.add_argument("--cache_data", action="store_true", help="Cache volumes in memory for fast epoch loops")
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, criterion, device, mode):
    model.train()
    total_loss_accum = 0.0
    seg_loss_accum = 0.0
    cls_loss_accum = 0.0
    tumor_loss_accum = 0.0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        seg_targets = batch['segmentation'].to(device) if mode in ['multitask', 'seg_only'] else None
        cls_targets = batch['classification'].to(device) if mode in ['multitask', 'cls_only'] else None
        
        optimizer.zero_grad()
        outputs = model(images)
        loss, loss_dict = criterion(outputs, seg_targets, cls_targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss_accum += loss.item()
        if 'loss_seg' in loss_dict:
            seg_loss_accum += loss_dict['loss_seg']
        if 'loss_cls' in loss_dict:
            cls_loss_accum += loss_dict['loss_cls']
        if 'loss_tumor_dice' in loss_dict:
            tumor_loss_accum += loss_dict['loss_tumor_dice']
            
    n = len(dataloader)
    return {
        'total_loss': total_loss_accum / n,
        'seg_loss': seg_loss_accum / n if n > 0 else 0,
        'cls_loss': cls_loss_accum / n if n > 0 else 0,
        'tumor_loss': tumor_loss_accum / n if n > 0 else 0
    }

def validate(model, dataloader, criterion, device, mode):
    model.eval()
    total_loss_accum = 0.0
    seg_dices_liver = []
    seg_dices_tumor = []
    cls_preds_all = []
    cls_targets_all = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            seg_targets = batch['segmentation'].to(device) if mode in ['multitask', 'seg_only'] else None
            cls_targets = batch['classification'].to(device) if mode in ['multitask', 'cls_only'] else None
            
            outputs = model(images)
            loss, _ = criterion(outputs, seg_targets, cls_targets)
            total_loss_accum += loss.item()
            
            if mode in ['multitask', 'seg_only']:
                seg_res = evaluate_segmentation_batch(outputs['seg_logits'], seg_targets)
                seg_dices_liver.append(seg_res['liver_dice'])
                seg_dices_tumor.append(seg_res['tumor_dice'])
                
            if mode in ['multitask', 'cls_only']:
                cls_preds_all.append(outputs['cls_logits'].cpu())
                cls_targets_all.append(batch['classification'].cpu())
                
    n = len(dataloader)
    val_metrics = {'val_loss': total_loss_accum / n}
    
    if mode in ['multitask', 'seg_only']:
        val_metrics['liver_dice'] = np.mean(seg_dices_liver)
        val_metrics['tumor_dice'] = np.mean(seg_dices_tumor)
        
    if mode in ['multitask', 'cls_only']:
        all_logits = torch.cat(cls_preds_all, dim=0)
        all_targets = torch.cat(cls_targets_all, dim=0)
        cls_res = evaluate_classification_batch(all_logits, all_targets)
        val_metrics['accuracy'] = cls_res['hamming_acc']
        val_metrics['subset_acc'] = cls_res['subset_acc']
        val_metrics['precision'] = cls_res['precision']
        val_metrics['recall'] = cls_res['recall']
        val_metrics['f1'] = cls_res['f1']
        val_metrics['per_class_f1'] = cls_res['per_class_f1']
        
    return val_metrics

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    os.makedirs(args.save_dir, exist_ok=True)
    
    target_shape = (args.target_size, args.target_size, args.target_size)
    
    # Setup Datasets
    if args.split_type == "3way":
        train_ds = LiMTMultiTaskDataset(
            data_dir=args.data_dir, splits_csv=args.splits_csv,
            split_type="3way", split_val="train",
            target_shape=target_shape, use_clahe=args.use_clahe,
            augment=True, cache_data=args.cache_data
        )
        val_ds = LiMTMultiTaskDataset(
            data_dir=args.data_dir, splits_csv=args.splits_csv,
            split_type="3way", split_val="val",
            target_shape=target_shape, use_clahe=args.use_clahe,
            augment=False, cache_data=args.cache_data
        )
        test_ds = LiMTMultiTaskDataset(
            data_dir=args.data_dir, splits_csv=args.splits_csv,
            split_type="3way", split_val="test",
            target_shape=target_shape, use_clahe=args.use_clahe,
            augment=False, cache_data=args.cache_data
        )
    else: # kfold
        train_ds = LiMTMultiTaskDataset(
            data_dir=args.data_dir, splits_csv=args.splits_csv,
            split_type="kfold", split_val=("train", args.fold),
            target_shape=target_shape, use_clahe=args.use_clahe,
            augment=True, cache_data=args.cache_data
        )
        val_ds = LiMTMultiTaskDataset(
            data_dir=args.data_dir, splits_csv=args.splits_csv,
            split_type="kfold", split_val=("val", args.fold),
            target_shape=target_shape, use_clahe=args.use_clahe,
            augment=False, cache_data=args.cache_data
        )
        test_ds = val_ds
        
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize Model
    if args.mode == "multitask":
        if args.model_arch == "msff":
            model = MultiTaskLiverNet_MSFF(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=16)
        else:
            model = MultiTaskLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=16)
    elif args.mode == "seg_only":
        model = SingleTaskSegmentationNet(in_channels=1, num_seg_classes=6, base_channels=16)
    else: # cls_only
        model = SingleTaskClassificationNet(in_channels=1, num_cls_classes=5, base_channels=16)
        
    model = model.to(device)
    criterion = TumorAwareMultiTaskLoss(
        lambda_seg=args.lambda_seg,
        lambda_cls=args.lambda_cls,
        beta_tumor=args.beta_tumor,
        device=device
    )
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

    if args.eval_only:
        print("\n--- Evaluating on Test Set ---")
        test_metrics = validate(model, test_loader, criterion, device, args.mode)
        print("Test Evaluation Results:")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k:15s}: {v:6.2f}")
            else:
                print(f"  {k:15s}: {v}")
        return test_metrics

    # Training
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_score = -float('inf')
    arch_tag = f"{args.model_arch}_" if args.mode == "multitask" else ""
    best_ckpt_path = os.path.join(args.save_dir, f"best_{arch_tag}{args.mode}_{args.split_type}.pth")
    
    print(f"\n--- Starting Training ({args.mode.upper()} / Arch: {args.model_arch}) for {args.epochs} Epochs ---")
    for epoch in range(1, args.epochs + 1):
        train_res = train_epoch(model, train_loader, optimizer, criterion, device, args.mode)
        val_res = validate(model, val_loader, criterion, device, args.mode)
        scheduler.step()
        
        # Scoring prioritizes tumor dice alongside liver dice and classification F1
        if args.mode == "multitask":
            score = val_res.get('liver_dice', 0) + 3.0 * val_res.get('tumor_dice', 0) + val_res.get('f1', 0)
            log_str = f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {train_res['total_loss']:.3f} | " \
                      f"Val Loss: {val_res['val_loss']:.3f} | Liver Dice: {val_res['liver_dice']:.1f}% | " \
                      f"Tumor Dice: {val_res['tumor_dice']:.1f}% | Cls Acc: {val_res['accuracy']:.1f}% | Cls F1: {val_res['f1']:.1f}%"
        elif args.mode == "seg_only":
            score = val_res.get('liver_dice', 0) + 3.0 * val_res.get('tumor_dice', 0)
            log_str = f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {train_res['total_loss']:.3f} | " \
                      f"Val Loss: {val_res['val_loss']:.3f} | Liver Dice: {val_res['liver_dice']:.1f}% | Tumor Dice: {val_res['tumor_dice']:.1f}%"
        else:
            score = val_res.get('f1', 0)
            log_str = f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {train_res['total_loss']:.3f} | " \
                      f"Val Loss: {val_res['val_loss']:.3f} | Cls Acc: {val_res['accuracy']:.1f}% | Cls F1: {val_res['f1']:.1f}%"
                      
        print(log_str)
        
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
    
    # Run test evaluation on best checkpoint
    print("\n--- Running Final Evaluation on Test Split ---")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    test_metrics = validate(model, test_loader, criterion, device, args.mode)
    print("Test Results with Best Model:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:6.2f}")
        else:
            print(f"  {k:15s}: {v}")

    return test_metrics

if __name__ == "__main__":
    main()
