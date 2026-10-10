import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)

from dataset import LiMTMultiTaskDataset
from models.multitask_net import MultiTaskLiverNet, SingleTaskClassificationNet
from models.multitask_msff_net import MultiTaskLiverNet_MSFF
from models.resvnet_fusion import ResVNetFusion
from models.weedsense_liver import WeedSenseLiverNet
from models.coseg_liver import CoSegLiverNet
from models.sopseg_liver import CoSeg_SOPSeg_LiverNet
from models.waveattn_sopseg_liver import WaveAttn_SOPSegLiverNet, CoSeg_WaveAttn_SOPSegLiverNet

CLASS_NAMES = ["Liver (0)", "Cyst (1)", "Hemangioma (2)", "Metastasis (3)", "HCC (4)"]

def load_model(model_name, ckpt_path, device):
    """Loads appropriate model architecture and weights."""
    if model_name == "cls_only":
        model = SingleTaskClassificationNet(in_channels=1, num_cls_classes=5, base_channels=16)
    elif model_name == "multitask_baseline":
        model = MultiTaskLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=16)
    elif model_name == "multitask_msff":
        model = MultiTaskLiverNet_MSFF(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=16)
    elif model_name in ["ref78_vanilla", "ref78_enhanced"]:
        model = ResVNetFusion(in_channels=1, num_seg_classes=2, num_cls_classes=5, base_channels=16)
    elif model_name == "weedsense":
        model = WeedSenseLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5)
    elif model_name == "coseg":
        model = CoSegLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=24)
    elif model_name == "coseg_sopseg":
        model = CoSeg_SOPSeg_LiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=24)
    elif model_name == "waveattn_sopseg":
        model = WaveAttn_SOPSegLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=24)
    elif model_name == "coseg_waveattn_sopseg":
        model = CoSeg_WaveAttn_SOPSegLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=24)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def extract_predictions(model, dataloader, device):
    """Extracts raw logits and probabilities for the entire dataset."""
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['classification']
            out = model(images)
            logits = out['cls_logits'].cpu()
            all_logits.append(logits)
            all_targets.append(targets)
            
    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy().astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs, targets

def compute_all_metrics(probs, targets, threshold=0.5):
    """Computes comprehensive classification scores."""
    preds_bin = (probs >= threshold).astype(int)
    
    # 1. Overall Metrics
    hamming_acc = (targets == preds_bin).mean() * 100.0
    subset_acc = accuracy_score(targets, preds_bin) * 100.0
    macro_prec = precision_score(targets, preds_bin, average='macro', zero_division=0) * 100.0
    macro_rec = recall_score(targets, preds_bin, average='macro', zero_division=0) * 100.0
    macro_f1 = f1_score(targets, preds_bin, average='macro', zero_division=0) * 100.0
    
    # Micro metrics
    micro_prec = precision_score(targets, preds_bin, average='micro', zero_division=0) * 100.0
    micro_rec = recall_score(targets, preds_bin, average='micro', zero_division=0) * 100.0
    micro_f1 = f1_score(targets, preds_bin, average='micro', zero_division=0) * 100.0
    
    # AUC-ROC (skip classes with only 1 unique value in targets if any)
    auc_scores = []
    for c in range(targets.shape[1]):
        if len(np.unique(targets[:, c])) > 1:
            try:
                auc_c = roc_auc_score(targets[:, c], probs[:, c]) * 100.0
                auc_scores.append(auc_c)
            except Exception:
                pass
    macro_auc = np.mean(auc_scores) if auc_scores else 0.0

    # 2. Per-class metrics
    per_cls_prec = precision_score(targets, preds_bin, average=None, zero_division=0) * 100.0
    per_cls_rec = recall_score(targets, preds_bin, average=None, zero_division=0) * 100.0
    per_cls_f1 = f1_score(targets, preds_bin, average=None, zero_division=0) * 100.0
    per_cls_acc = [(targets[:, c] == preds_bin[:, c]).mean() * 100.0 for c in range(5)]
    supports = targets.sum(axis=0)

    per_class_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Support": supports,
        "Accuracy (%)": [round(x, 2) for x in per_cls_acc],
        "Precision (%)": [round(x, 2) for x in per_cls_prec],
        "Recall (%)": [round(x, 2) for x in per_cls_rec],
        "F1 (%)": [round(x, 2) for x in per_cls_f1]
    })

    summary = {
        "Hamming Acc (%)": hamming_acc,
        "Subset Acc (%)": subset_acc,
        "Macro Precision (%)": macro_prec,
        "Macro Recall (%)": macro_rec,
        "Macro F1 (%)": macro_f1,
        "Micro F1 (%)": micro_f1,
        "Macro AUC-ROC (%)": macro_auc
    }
    
    return summary, per_class_df

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Define model configurations
    models_config = [
        {
            "id": "cls_only",
            "name": "Single-Task (Classification-Only 3D ResNet)",
            "ckpt": "checkpoints/best_cls_only_3way.pth",
            "target_size": 64,
            "hu_min": -100.0,
            "hu_max": 250.0,
            "clahe": False
        },
        {
            "id": "multitask_baseline",
            "name": "MultiTaskLiverNet (Initial Baseline)",
            "ckpt": "checkpoints/best_multitask_3way.pth",
            "target_size": 64,
            "hu_min": -100.0,
            "hu_max": 250.0,
            "clahe": False
        },
        {
            "id": "multitask_msff",
            "name": "MultiTaskLiverNet_MSFF (ECAdd + DFE + CLAHE)",
            "ckpt": "checkpoints/best_msff_multitask_3way.pth",
            "target_size": 80,
            "hu_min": -100.0,
            "hu_max": 250.0,
            "clahe": True
        },
        {
            "id": "ref78_vanilla",
            "name": "Ref [78] ResVNetFusion (Vanilla)",
            "ckpt": "checkpoints/best_resvnet_fusion_3way.pth",
            "target_size": 80,
            "hu_min": -200.0,
            "hu_max": 340.0,
            "clahe": True
        },
        {
            "id": "ref78_enhanced",
            "name": "Ref [78] ResVNetFusion (Enhanced Tumor Loss)",
            "ckpt": "checkpoints/best_ref78_enhanced.pth",
            "target_size": 80,
            "hu_min": -200.0,
            "hu_max": 340.0,
            "clahe": True
        },
        {
            "id": "weedsense",
            "name": "WeedSenseLiverNet",
            "ckpt": "checkpoints/best_weedsense_liver_3way.pth",
            "target_size": 64,
            "hu_min": -100.0,
            "hu_max": 250.0,
            "clahe": True
        },
        {
            "id": "coseg",
            "name": "Co-Seg (MICCAI 2025)",
            "ckpt": "checkpoints/best_coseg_liver_3way.pth",
            "target_size": 64,
            "hu_min": -100.0,
            "hu_max": 250.0,
            "clahe": True
        },
        {
            "id": "coseg_sopseg",
            "name": "Co-Seg + SOPSeg (Combined)",
            "ckpt": "checkpoints/best_coseg_sopseg_3way.pth",
            "target_size": 64,
            "hu_min": -100.0,
            "hu_max": 250.0,
            "clahe": True
        },
        {
            "id": "waveattn_sopseg",
            "name": "WaveAttn-SOPSeg (Standalone)",
            "ckpt": "checkpoints/best_waveattn_sopseg_3way.pth",
            "target_size": 64,
            "hu_min": -100.0,
            "hu_max": 250.0,
            "clahe": True
        },
        {
            "id": "coseg_waveattn_sopseg",
            "name": "Co-Seg + WaveAttn-SOPSeg",
            "ckpt": "checkpoints/best_coseg_waveattn_sopseg_3way.pth",
            "target_size": 64,
            "hu_min": -100.0,
            "hu_max": 250.0,
            "clahe": True
        }
    ]

    all_summaries = []
    all_per_class = {}
    threshold_sweeps = {}

    for cfg in models_config:
        ckpt_path = cfg["ckpt"]
        if not os.path.exists(ckpt_path):
            print(f"Skipping {cfg['name']} (checkpoint not found: {ckpt_path})")
            continue

        print(f"\n========================================================")
        print(f"  Evaluating: {cfg['name']}")
        print(f"========================================================")
        
        # Load dataset
        test_ds = LiMTMultiTaskDataset(
            data_dir="LiMT Dataset",
            splits_csv="LiMT Dataset/limt_splits.csv",
            split_type="3way",
            split_val="test",
            target_shape=(cfg["target_size"], cfg["target_size"], cfg["target_size"]),
            hu_min=cfg["hu_min"],
            hu_max=cfg["hu_max"],
            use_clahe=cfg["clahe"],
            augment=False,
            cache_data=True
        )
        test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)
        
        # Load model and run inference
        model = load_model(cfg["id"], ckpt_path, device)
        probs, targets = extract_predictions(model, test_loader, device)

        # Standard threshold (0.5)
        summary, per_class_df = compute_all_metrics(probs, targets, threshold=0.5)
        summary["Model"] = cfg["name"]
        all_summaries.append(summary)
        all_per_class[cfg["id"]] = per_class_df
        
        print("\n--- Overall Test Classification Metrics (Threshold=0.5) ---")
        for k, v in summary.items():
            if k != "Model":
                print(f"  {k:22s}: {v:6.2f}")
                
        print("\n--- Per-Class Breakdown ---")
        print(per_class_df.to_string(index=False))

        # Threshold sweep for best F1
        th_results = []
        for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            s_th, _ = compute_all_metrics(probs, targets, threshold=th)
            th_results.append({
                "Threshold": th,
                "Accuracy (%)": round(s_th["Hamming Acc (%)"], 2),
                "Subset Acc (%)": round(s_th["Subset Acc (%)"], 2),
                "Precision (%)": round(s_th["Macro Precision (%)"], 2),
                "Recall (%)": round(s_th["Macro Recall (%)"], 2),
                "Macro F1 (%)": round(s_th["Macro F1 (%)"], 2),
                "Micro F1 (%)": round(s_th["Micro F1 (%)"], 2),
            })
        threshold_sweeps[cfg["id"]] = pd.DataFrame(th_results)

    # Compile Overall Summary Table
    df_overall = pd.DataFrame(all_summaries)
    cols = ["Model", "Hamming Acc (%)", "Subset Acc (%)", "Macro Precision (%)", "Macro Recall (%)", "Macro F1 (%)", "Micro F1 (%)", "Macro AUC-ROC (%)"]
    df_overall = df_overall[[c for c in cols if c in df_overall.columns]]
    
    os.makedirs("results", exist_ok=True)
    df_overall.to_csv("results/classification_models_comparison.csv", index=False)
    
    print("\n\n========================================================")
    print("      FINAL CLASSIFICATION BENCHMARK COMPARISON         ")
    print("========================================================")
    print(df_overall.to_string(index=False))

    # Save per-class breakdown
    for m_id, df_cls in all_per_class.items():
        df_cls.to_csv(f"results/per_class_{m_id}.csv", index=False)

    # Save threshold sweeps
    for m_id, df_th in threshold_sweeps.items():
        df_th.to_csv(f"results/threshold_sweep_{m_id}.csv", index=False)

if __name__ == '__main__':
    main()
