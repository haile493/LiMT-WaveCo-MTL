import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_binary_dice_jaccard(pred_bin, target_bin, smooth=1e-5):
    """Computes Dice [%] and Jaccard [%] for binary masks."""
    pred_flat = pred_bin.flatten().astype(bool)
    target_flat = target_bin.flatten().astype(bool)
    
    intersection = np.logical_and(pred_flat, target_flat).sum()
    cardinality = pred_flat.sum() + target_flat.sum()
    union = np.logical_or(pred_flat, target_flat).sum()
    
    if cardinality == 0:
        dice = 100.0 if target_flat.sum() == 0 else 0.0
    else:
        dice = (2.0 * intersection + smooth) / (cardinality + smooth) * 100.0
        
    if union == 0:
        jaccard = 100.0 if target_flat.sum() == 0 else 0.0
    else:
        jaccard = (intersection + smooth) / (union + smooth) * 100.0
        
    return float(dice), float(jaccard)

def evaluate_segmentation_batch(pred_logits, targets, tumor_thresh=0.5):
    """
    pred_logits: torch.Tensor of shape (B, C, D, H, W)
      - C == 2: Channel 0 is Liver, Channel 1 is Tumor (Ref [78] dual-head)
      - C == 6: Multi-class (0: bg, 1: liver, 2..5: tumor subtypes)
      - C == 1: Binary Tumor only
    targets: torch.Tensor of shape (B, D, H, W) with integer indices 0..5
    """
    batch_size = pred_logits.shape[0]
    num_channels = pred_logits.shape[1]
    targets_np = targets.detach().cpu().numpy()
    
    if num_channels == 2:
        # Ref [78] dual-head
        probs = torch.sigmoid(pred_logits).detach().cpu().numpy()
        pred_liver_all = (probs[:, 0] >= 0.5)
        pred_tumor_all = (probs[:, 1] >= tumor_thresh)
    elif num_channels == 6:
        preds = torch.argmax(pred_logits, dim=1).detach().cpu().numpy()
        pred_liver_all = (preds >= 1)
        pred_tumor_all = (preds >= 2)
    else:
        probs = torch.sigmoid(pred_logits).detach().cpu().numpy()
        pred_liver_all = None
        pred_tumor_all = (probs[:, 0] >= tumor_thresh)
        
    liver_dices, liver_jaccards = [], []
    tumor_dices, tumor_jaccards = [], []
    
    total_tumor_intersection = 0
    total_tumor_cardinality = 0
    
    for i in range(batch_size):
        t = targets_np[i]
        
        # Liver (label >= 1)
        if pred_liver_all is not None:
            l_d, l_j = compute_binary_dice_jaccard(pred_liver_all[i], t >= 1)
            liver_dices.append(l_d)
            liver_jaccards.append(l_j)
            
        # Tumor (label >= 2)
        p_tum = pred_tumor_all[i]
        t_tum = (t >= 2)
        if t_tum.sum() > 0:
            t_d, t_j = compute_binary_dice_jaccard(p_tum, t_tum)
            tumor_dices.append(t_d)
            tumor_jaccards.append(t_j)
            
        total_tumor_intersection += np.logical_and(p_tum, t_tum).sum()
        total_tumor_cardinality += p_tum.sum() + t_tum.sum()
            
    global_tumor_dice = 0.0
    if total_tumor_cardinality > 0:
        global_tumor_dice = float((2.0 * total_tumor_intersection + 1e-5) / (total_tumor_cardinality + 1e-5) * 100.0)
        
    return {
        'liver_dice': np.mean(liver_dices) if liver_dices else 0.0,
        'liver_jaccard': np.mean(liver_jaccards) if liver_jaccards else 0.0,
        'tumor_dice': np.mean(tumor_dices) if tumor_dices else 0.0,
        'tumor_jaccard': np.mean(tumor_jaccards) if tumor_jaccards else 0.0,
        'global_tumor_dice': global_tumor_dice
    }

def evaluate_classification_batch(cls_logits, cls_targets, threshold=0.5):
    probs = torch.sigmoid(cls_logits).detach().cpu().numpy()
    preds_bin = (probs >= threshold).astype(int)
    targets_np = cls_targets.detach().cpu().numpy().astype(int)
    
    subset_acc = accuracy_score(targets_np, preds_bin) * 100.0
    hamming_acc = (targets_np == preds_bin).mean() * 100.0
    prec = precision_score(targets_np, preds_bin, average='macro', zero_division=0) * 100.0
    rec = recall_score(targets_np, preds_bin, average='macro', zero_division=0) * 100.0
    f1 = f1_score(targets_np, preds_bin, average='macro', zero_division=0) * 100.0
    per_class_f1 = f1_score(targets_np, preds_bin, average=None, zero_division=0) * 100.0
    
    return {
        'subset_acc': float(subset_acc),
        'hamming_acc': float(hamming_acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'per_class_f1': [float(x) for x in per_class_f1],
        'preds': preds_bin,
        'targets': targets_np
    }
