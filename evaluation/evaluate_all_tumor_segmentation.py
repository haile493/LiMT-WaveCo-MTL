"""
Comprehensive Liver and Tumor Segmentation Evaluation across all models on LiMT Test Set (30 cases).

Models compared:
1. Single-Task (Seg-Only 3D U-Net)
2. MultiTaskLiverNet (Initial Baseline MTL)
3. MultiTaskLiverNet_MSFF (Stage 1 MSFF / EDNet)
4. ResVNetFusion (Enhanced Ref [78])
5. WeedSenseLiverNet (Proposed 3D Adaptation of WeedSense)

Computes:
- Liver Dice & Jaccard (%)
- Tumor Dice & Jaccard (%)
- Tumor Sensitivity / Recall (%)
- Tumor Precision (%)
- Per-class Lesion Dice: Cyst, Hemangioma, Metastasis, HCC
- Generates a high-quality visualization panel saving to results/tumor_segmentation_visual.png
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from dataset import LiMTMultiTaskDataset
from metrics import compute_binary_dice_jaccard
from models.multitask_net import SingleTaskSegmentationNet, MultiTaskLiverNet
from models.multitask_msff_net import MultiTaskLiverNet_MSFF
from models.resvnet_fusion import ResVNetFusion
from models.weedsense_liver import WeedSenseLiverNet
from models.coseg_liver import CoSegLiverNet
from models.sopseg_liver import CoSeg_SOPSeg_LiverNet
from models.waveattn_sopseg_liver import WaveAttn_SOPSegLiverNet, CoSeg_WaveAttn_SOPSegLiverNet


def load_model(name, ckpt_path, device):
    if name == 'Single-Task Seg-Only':
        model = SingleTaskSegmentationNet(in_channels=1, num_seg_classes=6, base_channels=16)
    elif name == 'MultiTaskLiverNet (Baseline)':
        model = MultiTaskLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=16)
    elif name == 'MultiTaskLiverNet_MSFF':
        model = MultiTaskLiverNet_MSFF(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=16)
    elif 'ResVNetFusion' in name:
        model = ResVNetFusion(in_channels=1, num_seg_classes=2, num_cls_classes=5, base_channels=16)
    elif name == 'WeedSenseLiverNet' or name == 'WeedSenseLiverNet (Ours)':
        model = WeedSenseLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5)
    elif name == 'Co-Seg (MICCAI 2025)':
        model = CoSegLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=24)
    elif name == 'Co-Seg + SOPSeg (Combined)':
        model = CoSeg_SOPSeg_LiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=24)
    elif name == 'WaveAttn-SOPSeg (Standalone)':
        model = WaveAttn_SOPSegLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=24)
    elif name == 'Co-Seg + WaveAttn-SOPSeg':
        model = CoSeg_WaveAttn_SOPSegLiverNet(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=24)
    else:
        raise ValueError(f"Unknown model name: {name}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model_segmentation(model_name, model, dataloader, device, tumor_th=0.5):
    liver_dices = []
    liver_jaccards = []
    tumor_dices = []
    tumor_jaccards = []
    tumor_sensitivities = []
    tumor_precisions = []

    per_class_dices = {2: [], 3: [], 4: [], 5: []}  # Cyst, Hemangioma, Met, HCC

    all_cases_data = []

    for batch in dataloader:
        images = batch['image'].to(device)
        targets = batch['segmentation'].cpu().numpy()
        filenames = batch['filename']

        out = model(images)
        seg_logits = out['seg_logits'] if isinstance(out, dict) else out

        num_classes = seg_logits.shape[1]

        if num_classes == 2:
            # Dual-head (Channel 0: Liver, Channel 1: Tumor)
            probs = torch.sigmoid(seg_logits).cpu().numpy()
            pred_liver = (probs[:, 0] >= 0.5)
            pred_tumor = (probs[:, 1] >= tumor_th) & pred_liver
            preds_multi = None
        else:
            # Multi-class (0: bg, 1: liver, 2..5: lesions)
            probs = F.softmax(seg_logits, dim=1).cpu().numpy()
            preds_argmax = np.argmax(probs, axis=1)
            pred_liver = (preds_argmax >= 1)
            
            # Probability-based tumor detection within liver
            prob_tumor = probs[:, 2:].sum(axis=1)
            pred_tumor = (preds_argmax >= 2) | ((prob_tumor >= tumor_th) & pred_liver)
            preds_multi = preds_argmax

        B = images.shape[0]
        for b in range(B):
            t = targets[b]
            fname = filenames[b]
            t_liver = (t >= 1)
            t_tumor = (t >= 2)
            p_liv = pred_liver[b]
            p_tum = pred_tumor[b]

            l_dice, l_jacc = compute_binary_dice_jaccard(p_liv, t_liver)
            liver_dices.append(l_dice)
            liver_jaccards.append(l_jacc)

            has_tumor = (t_tumor.sum() > 0)
            if has_tumor:
                t_dice, t_jacc = compute_binary_dice_jaccard(p_tum, t_tumor)
                tumor_dices.append(t_dice)
                tumor_jaccards.append(t_jacc)

                # Sensitivity / Recall
                inter = np.logical_and(p_tum, t_tumor).sum()
                sens = (inter / t_tumor.sum() * 100.0) if t_tumor.sum() > 0 else 0.0
                prec = (inter / p_tum.sum() * 100.0) if p_tum.sum() > 0 else 0.0
                tumor_sensitivities.append(sens)
                tumor_precisions.append(prec)

                # Subtype Dices (for 6-class models)
                if preds_multi is not None:
                    for c in [2, 3, 4, 5]:
                        t_c = (t == c)
                        p_c = (preds_multi[b] == c)
                        if t_c.sum() > 0:
                            d_c, _ = compute_binary_dice_jaccard(p_c, t_c)
                            per_class_dices[c].append(d_c)
            else:
                # Normal liver without tumor: check false positives
                fp = p_tum.sum()
                t_dice = 100.0 if fp == 0 else 0.0

            all_cases_data.append({
                'case': fname,
                'has_tumor': has_tumor,
                'liver_dice': l_dice,
                'tumor_dice': t_dice if has_tumor else np.nan
            })

    return {
        'model': model_name,
        'liver_dice': float(np.mean(liver_dices)),
        'liver_jaccard': float(np.mean(liver_jaccards)),
        'tumor_dice': float(np.mean(tumor_dices)) if len(tumor_dices) > 0 else 0.0,
        'tumor_jaccard': float(np.mean(tumor_jaccards)) if len(tumor_jaccards) > 0 else 0.0,
        'tumor_sensitivity': float(np.mean(tumor_sensitivities)) if len(tumor_sensitivities) > 0 else 0.0,
        'tumor_precision': float(np.mean(tumor_precisions)) if len(tumor_precisions) > 0 else 0.0,
        'cyst_dice': float(np.mean(per_class_dices[2])) if len(per_class_dices[2]) > 0 else 0.0,
        'hemangioma_dice': float(np.mean(per_class_dices[3])) if len(per_class_dices[3]) > 0 else 0.0,
        'metastasis_dice': float(np.mean(per_class_dices[4])) if len(per_class_dices[4]) > 0 else 0.0,
        'hcc_dice': float(np.mean(per_class_dices[5])) if len(per_class_dices[5]) > 0 else 0.0,
        'cases': all_cases_data
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Liver Tumor Segmentation Evaluation on {device}...")

    MODELS_CONFIG = [
        ('Single-Task Seg-Only', 'checkpoints/best_seg_only_3way.pth'),
        ('MultiTaskLiverNet (Baseline)', 'checkpoints/best_multitask_3way.pth'),
        ('MultiTaskLiverNet_MSFF', 'checkpoints/best_msff_multitask_3way.pth'),
        ('ResVNetFusion', 'checkpoints/best_ref78_enhanced.pth'),
        ('WeedSenseLiverNet', 'checkpoints/best_weedsense_liver_3way.pth'),
        ('Co-Seg (MICCAI 2025)', 'checkpoints/best_coseg_liver_3way.pth'),
        ('Co-Seg + SOPSeg (Combined)', 'checkpoints/best_coseg_sopseg_3way.pth'),
        ('WaveAttn-SOPSeg (Standalone)', 'checkpoints/best_waveattn_sopseg_3way.pth'),
        ('Co-Seg + WaveAttn-SOPSeg', 'checkpoints/best_coseg_waveattn_sopseg_3way.pth'),
    ]

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
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    results_table = []
    loaded_models = {}

    for name, ckpt_path in MODELS_CONFIG:
        print(f"\nEvaluating: {name}...")
        model = load_model(name, ckpt_path, device)
        loaded_models[name] = model

        # Calibrated threshold evaluation
        res = evaluate_model_segmentation(name, model, test_loader, device, tumor_th=0.45)
        results_table.append(res)

        print(f"  Liver Dice:          {res['liver_dice']:.2f}% (Jaccard: {res['liver_jaccard']:.2f}%)")
        print(f"  Tumor Dice:          {res['tumor_dice']:.2f}% (Jaccard: {res['tumor_jaccard']:.2f}%)")
        print(f"  Tumor Sensitivity:   {res['tumor_sensitivity']:.2f}%")
        print(f"  Tumor Precision:     {res['tumor_precision']:.2f}%")
        if res['cyst_dice'] > 0 or res['hemangioma_dice'] > 0:
            print(f"  Subtype Dices -> Cyst: {res['cyst_dice']:.2f}%, Hemangioma: {res['hemangioma_dice']:.2f}%, Met: {res['metastasis_dice']:.2f}%, HCC: {res['hcc_dice']:.2f}%")

    # Format Summary Table
    df_summary = pd.DataFrame([{
        'Model': r['model'],
        'Liver Dice (%)': round(r['liver_dice'], 2),
        'Liver Jaccard (%)': round(r['liver_jaccard'], 2),
        'Tumor Dice (%)': round(r['tumor_dice'], 2),
        'Tumor Jaccard (%)': round(r['tumor_jaccard'], 2),
        'Tumor Sensitivity (%)': round(r['tumor_sensitivity'], 2),
        'Tumor Precision (%)': round(r['tumor_precision'], 2)
    } for r in results_table])

    os.makedirs('results', exist_ok=True)
    df_summary.to_csv('results/liver_tumor_segmentation_comparison.csv', index=False)

    print("\n" + "=" * 90)
    print("LIVER TUMOR SEGMENTATION BENCHMARK SUMMARY (LiMT Test Set, 30 Cases)")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)

    # Generate Qualitative Visualizations
    generate_visualizations(loaded_models, test_loader, device)


def generate_visualizations(models_dict, dataloader, device):
    """Generates slice-by-slice visual comparison for a representative tumor case."""
    print("\nGenerating qualitative visualization comparison...")

    # Find a test case with a distinct tumor
    sample_img, sample_target, sample_name = None, None, None
    for batch in dataloader:
        imgs = batch['image']
        targets = batch['segmentation']
        fnames = batch['filename']
        for b in range(imgs.shape[0]):
            t = targets[b]
            if (t >= 2).sum() > 200:  # Has significant tumor
                sample_img = imgs[b:b+1].to(device)
                sample_target = t.numpy()
                sample_name = fnames[b]
                break
        if sample_img is not None:
            break

    if sample_img is None:
        print("No suitable tumor case found for visualization.")
        return

    # Find the axial slice with the largest tumor area
    tumor_mask_3d = (sample_target >= 2)
    slice_idx = int(np.argmax(tumor_mask_3d.sum(axis=(1, 2))))

    ct_slice = sample_img[0, 0, slice_idx].cpu().numpy()
    gt_slice = sample_target[slice_idx]

    preds_dict = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            out = model(sample_img)
            logits = out['seg_logits'] if isinstance(out, dict) else out
            if logits.shape[1] == 2:
                prob = torch.sigmoid(logits).cpu().numpy()[0, :, slice_idx]
                p_liv = prob[0] >= 0.5
                p_tum = (prob[1] >= 0.45) & p_liv
                pred_map = np.zeros_like(gt_slice)
                pred_map[p_liv] = 1
                pred_map[p_tum] = 2
            else:
                p_argmax = np.argmax(logits[0].cpu().numpy(), axis=0)[slice_idx]
                pred_map = p_argmax
            preds_dict[name] = pred_map

    # Plot
    fig, axes = plt.subplots(1, 2 + len(models_dict), figsize=(20, 4))
    
    # 1. CT Slice
    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title(f"CT Slice #{slice_idx}\n({sample_name})", fontsize=10)
    axes[0].axis('off')

    # 2. Ground Truth
    axes[1].imshow(ct_slice, cmap='gray')
    gt_overlay = np.ma.masked_where(gt_slice == 0, gt_slice)
    axes[1].imshow(gt_overlay, cmap='jet', alpha=0.6, vmin=0, vmax=5)
    axes[1].set_title("Ground Truth\n(Liver + Tumor)", fontsize=10)
    axes[1].axis('off')

    # Model predictions
    for idx, (name, p_map) in enumerate(preds_dict.items()):
        ax = axes[2 + idx]
        ax.imshow(ct_slice, cmap='gray')
        p_overlay = np.ma.masked_where(p_map == 0, p_map)
        ax.imshow(p_overlay, cmap='jet', alpha=0.6, vmin=0, vmax=5)
        short_name = name.replace(" (Baseline)", "").replace(" (Ours)", "").replace("Single-Task ", "")
        ax.set_title(f"{short_name}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    out_img_path = 'results/tumor_segmentation_comparison.png'
    plt.savefig(out_img_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved qualitative visualization to {out_img_path}")


if __name__ == '__main__':
    main()
