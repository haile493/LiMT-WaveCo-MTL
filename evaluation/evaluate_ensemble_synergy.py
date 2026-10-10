"""
Full 3D Volumetric Evaluation of WeakMedSAM Synergy Ensemble across all 30 LiMT Test Cases.
Combines:
- WeakMedSAM-Seg: High-resolution in-plane tumor detection and feature adapters
- WeakMedSAM + Co-Seg + WaveAttn: 2.5D multi-slice [z-1, z, z+1] continuity, Haar DWT edge attention, and SOPSeg texture covariance
Computes complete 3D metrics (Dice, Jaccard, ASD, HD95, Sens, Prec) and saves benchmark comparison.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure, label
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.weakmedsam_seg_net import WeakMedSAM_SegNet
from models.weakmedsam_coseg_waveattn import WeakMedSAM_CoSeg_WaveAttn_Net
from evaluate_weakmedsam_limt import postprocess_3d_liver, postprocess_3d_tumor, compute_binary_metrics_3d, compute_asd_hd95, window_slice
from evaluate_weakmedsam_combined import window_25d_slice

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running WeakMedSAM Synergy Ensemble 3D Evaluation on {device} ===")

    os.makedirs("results", exist_ok=True)
    splits_csv = "LiMT Dataset/limt_splits.csv"
    df = pd.read_csv(splits_csv)
    test_df = df[df["split_3way"] == "test"].copy().reset_index(drop=True)
    print(f"Total test cases to evaluate: {len(test_df)}")

    # Load Model 1: WeakMedSAM-Seg
    ckpt1 = "checkpoints/best_weakmedsam_seg_limt.pth"
    m1 = WeakMedSAM_SegNet(pretrained=False)
    state1 = torch.load(ckpt1, map_location=device, weights_only=False)
    m1.load_state_dict(state1["model_state_dict"] if "model_state_dict" in state1 else state1)
    m1.to(device).eval()
    print(f"Loaded Model 1 (WeakMedSAM-Seg) from {ckpt1}")

    # Load Model 2: WeakMedSAM + Co-Seg + WaveAttn
    ckpt2 = "checkpoints/best_weakmedsam_coseg_waveattn.pth"
    m2 = WeakMedSAM_CoSeg_WaveAttn_Net(pretrained=False)
    state2 = torch.load(ckpt2, map_location=device, weights_only=False)
    m2.load_state_dict(state2["model_state_dict"] if "model_state_dict" in state2 else state2)
    m2.to(device).eval()
    print(f"Loaded Model 2 (WeakMedSAM + Co-Seg + WaveAttn) from {ckpt2}")

    case_records = []
    liver_dices, liver_jaccs, liver_asds, liver_hd95s = [], [], [], []
    tumor_dices, tumor_jaccs, tumor_asds, tumor_hd95s = [], [], [], []
    tumor_sens_list, tumor_prec_list = [], []

    sample_visuals = []

    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Ensemble 3D Evaluation"):
            fname = row["Filename"]
            vol_path = os.path.join("LiMT Dataset/ct", fname)
            seg_fname = fname.replace("volume_", "segmentation_")
            seg_path = os.path.join("LiMT Dataset/seg", seg_fname)

            vol_obj = nib.load(vol_path)
            vol = vol_obj.get_fdata()
            zooms = vol_obj.header.get_zooms()[:3]
            seg = nib.load(seg_path).get_fdata()

            H, W, D = vol.shape
            pred_liver_3d = np.zeros((H, W, D), dtype=np.uint8)
            pred_tumor_3d = np.zeros((H, W, D), dtype=np.uint8)

            best_slice_idx = 0
            max_tumor_pix = 0

            for z in range(D):
                z_prev = max(0, z - 1)
                z_next = min(D - 1, z + 1)

                slice_p = vol[:, :, z_prev]
                slice_c = vol[:, :, z]
                slice_n = vol[:, :, z_next]
                gt_slice_seg = seg[:, :, z]

                t_pix = (gt_slice_seg >= 2).sum()
                if t_pix > max_tumor_pix:
                    max_tumor_pix = t_pix
                    best_slice_idx = z

                # 256x256 resizing
                sc_256 = cv2.resize(slice_c, (256, 256), interpolation=cv2.INTER_LINEAR)
                sp_256 = cv2.resize(slice_p, (256, 256), interpolation=cv2.INTER_LINEAR)
                sn_256 = cv2.resize(slice_n, (256, 256), interpolation=cv2.INTER_LINEAR)

                inp1 = window_slice(sc_256).unsqueeze(0).to(device)
                inp2 = window_25d_slice(sp_256, sc_256, sn_256).unsqueeze(0).to(device)

                with torch.amp.autocast('cuda'):
                    out1 = m1(inp1)
                    out2 = m2(inp2)

                p1 = torch.sigmoid(out1["seg_logits"].float()).squeeze(0).cpu().numpy().astype(np.float32)
                p2 = torch.sigmoid(out2["seg_logits"].float()).squeeze(0).cpu().numpy().astype(np.float32)

                # Synergistic Probability Fusion:
                # Model 2 provides superior liver boundary continuity (ASD 1.55mm)
                # Model 1 provides superior tumor precision and sensitivity
                p_liv_fuse = 0.4 * p1[0] + 0.6 * p2[0]
                p_tum_fuse = 0.65 * p1[1] + 0.35 * p2[1]

                # Resize to original (W, H)
                pliv_orig = cv2.resize(p_liv_fuse, (W, H), interpolation=cv2.INTER_LINEAR)
                ptum_orig = cv2.resize(p_tum_fuse, (W, H), interpolation=cv2.INTER_LINEAR)

                p_liv_bin = (pliv_orig >= 0.50).astype(np.uint8)
                p_tum_bin = ((ptum_orig >= 0.50) & (pliv_orig >= 0.35)).astype(np.uint8)

                pred_liver_3d[:, :, z] = p_liv_bin
                pred_tumor_3d[:, :, z] = p_tum_bin

            # 3D Anatomical Cleaning
            pred_liver_3d = postprocess_3d_liver(pred_liver_3d)
            pred_tumor_3d = postprocess_3d_tumor(pred_tumor_3d, pred_liver_3d)

            gt_liver_3d = (seg >= 1).astype(np.uint8)
            gt_tumor_3d = (seg >= 2).astype(np.uint8)

            l_dice, l_jacc, _, _ = compute_binary_metrics_3d(pred_liver_3d, gt_liver_3d)
            l_asd, l_hd95 = compute_asd_hd95(pred_liver_3d, gt_liver_3d, zooms)

            liver_dices.append(l_dice)
            liver_jaccs.append(l_jacc)
            liver_asds.append(l_asd)
            liver_hd95s.append(l_hd95)

            has_tumor_3d = (gt_tumor_3d.sum() > 0)
            if has_tumor_3d:
                t_dice, t_jacc, t_sens, t_prec = compute_binary_metrics_3d(pred_tumor_3d, gt_tumor_3d)
                t_asd, t_hd95 = compute_asd_hd95(pred_tumor_3d, gt_tumor_3d, zooms)
                tumor_dices.append(t_dice)
                tumor_jaccs.append(t_jacc)
                tumor_sens_list.append(t_sens)
                tumor_prec_list.append(t_prec)
                tumor_asds.append(t_asd)
                tumor_hd95s.append(t_hd95)
            else:
                fp = pred_tumor_3d.sum()
                t_dice = 100.0 if fp == 0 else 0.0
                t_jacc = 100.0 if fp == 0 else 0.0
                t_sens = 100.0
                t_prec = 100.0 if fp == 0 else 0.0
                t_asd, t_hd95 = 0.0, 0.0

            case_records.append({
                "case": fname,
                "has_tumor": has_tumor_3d,
                "liver_dice": l_dice,
                "liver_jaccard": l_jacc,
                "liver_asd_mm": l_asd,
                "liver_hd95_mm": l_hd95,
                "tumor_dice": t_dice,
                "tumor_jaccard": t_jacc,
                "tumor_sensitivity": t_sens if has_tumor_3d else np.nan,
                "tumor_precision": t_prec if has_tumor_3d else np.nan,
                "tumor_asd_mm": t_asd if has_tumor_3d else np.nan,
                "tumor_hd95_mm": t_hd95 if has_tumor_3d else np.nan,
            })

            if has_tumor_3d and len(sample_visuals) < 4:
                sample_visuals.append({
                    "case": fname,
                    "slice_idx": best_slice_idx,
                    "ct": vol[:, :, best_slice_idx],
                    "gt_liver": gt_liver_3d[:, :, best_slice_idx],
                    "gt_tumor": gt_tumor_3d[:, :, best_slice_idx],
                    "pred_liver": pred_liver_3d[:, :, best_slice_idx],
                    "pred_tumor": pred_tumor_3d[:, :, best_slice_idx],
                    "dice_liv": l_dice,
                    "dice_tum": t_dice
                })

    df_cases = pd.DataFrame(case_records)
    case_csv_path = "results/ensemble_synergy_per_case_test_results.csv"
    df_cases.to_csv(case_csv_path, index=False)

    mLivDice = np.mean(liver_dices)
    sLivDice = np.std(liver_dices)
    mLivJacc = np.mean(liver_jaccs)
    sLivJacc = np.std(liver_jaccs)
    mLivASD = np.mean(liver_asds)
    mLivHD95 = np.mean(liver_hd95s)

    mTumDice = np.mean(tumor_dices)
    sTumDice = np.std(tumor_dices)
    mTumJacc = np.mean(tumor_jaccs)
    sTumJacc = np.std(tumor_jaccs)
    mTumSens = np.mean(tumor_sens_list)
    mTumPrec = np.mean(tumor_prec_list)
    mTumASD = np.mean(tumor_asds)
    mTumHD95 = np.mean(tumor_hd95s)

    print("\n" + "="*80)
    print("WEAKMEDSAM SYNERGY ENSEMBLE SUMMARY ON LIMT TEST SET (30 CASES)")
    print("="*80)
    print(f"LIVER SEGMENTATION:")
    print(f"  Dice (%):      {mLivDice:.2f} +/- {sLivDice:.2f}")
    print(f"  Jaccard (%):   {mLivJacc:.2f} +/- {sLivJacc:.2f}")
    print(f"  ASD (mm):      {mLivASD:.2f}")
    print(f"  HD95 (mm):     {mLivHD95:.2f}")
    print(f"\nTUMOR SEGMENTATION (Tumor-bearing Cases):")
    print(f"  Dice (%):      {mTumDice:.2f} +/- {sTumDice:.2f}")
    print(f"  Jaccard (%):   {mTumJacc:.2f} +/- {sTumJacc:.2f}")
    print(f"  Sensitivity:   {mTumSens:.2f}%")
    print(f"  Precision:     {mTumPrec:.2f}%")
    print(f"  ASD (mm):      {mTumASD:.2f}")
    print(f"  HD95 (mm):     {mTumHD95:.2f}")
    print("="*80)

    benchmarks = [
        {"Method": "3D U-Net (LiMT Paper)", "Liver Dice": 96.48, "Liver Jaccard": 92.19, "Liver ASD": 4.48, "Liver HD95": 1.56, "Tumor Dice": 55.91, "Tumor Jaccard": 38.78, "Tumor ASD": 9.49, "Tumor HD95": 36.37},
        {"Method": "V-Net (LiMT Paper)", "Liver Dice": 96.43, "Liver Jaccard": 91.87, "Liver ASD": 3.88, "Liver HD95": 1.48, "Tumor Dice": 54.80, "Tumor Jaccard": 36.15, "Tumor ASD": 7.08, "Tumor HD95": 33.50},
        {"Method": "Swin Transformer (LiMT Paper)", "Liver Dice": 96.64, "Liver Jaccard": 92.35, "Liver ASD": 5.78, "Liver HD95": 2.01, "Tumor Dice": 50.44, "Tumor Jaccard": 35.83, "Tumor ASD": 9.29, "Tumor HD95": 35.57},
        {"Method": "nn-UNet (LiMT Paper)", "Liver Dice": 97.03, "Liver Jaccard": 93.28, "Liver ASD": 3.93, "Liver HD95": 0.43, "Tumor Dice": 58.80, "Tumor Jaccard": 40.73, "Tumor ASD": 6.57, "Tumor HD95": 30.85},
        {"Method": "Multi-Task [78] (LiMT Paper)", "Liver Dice": 97.03, "Liver Jaccard": 93.28, "Liver ASD": 3.93, "Liver HD95": 0.43, "Tumor Dice": 60.04, "Tumor Jaccard": 43.58, "Tumor ASD": 5.95, "Tumor HD95": 28.73},
        {"Method": "WeakMedSAM-Seg (Ours)", "Liver Dice": 94.00, "Liver Jaccard": 88.87, "Liver ASD": 1.75, "Liver HD95": 8.03, "Tumor Dice": 66.60, "Tumor Jaccard": 48.22, "Tumor ASD": 19.50, "Tumor HD95": 57.30},
        {"Method": "WeakMedSAM+CoSeg+WaveAttn", "Liver Dice": 93.92, "Liver Jaccard": 88.91, "Liver ASD": 1.55, "Liver HD95": 7.29, "Tumor Dice": 37.61, "Tumor Jaccard": 27.30, "Tumor ASD": 30.09, "Tumor HD95": 66.71},
        {"Method": "WeakMedSAM Synergy Ensemble", "Liver Dice": mLivDice, "Liver Jaccard": mLivJacc, "Liver ASD": mLivASD, "Liver HD95": mLivHD95, "Tumor Dice": mTumDice, "Tumor Jaccard": mTumJacc, "Tumor ASD": mTumASD, "Tumor HD95": mTumHD95},
    ]

    df_bench = pd.DataFrame(benchmarks)
    bench_csv = "results/ensemble_synergy_benchmark_comparison.csv"
    df_bench.to_csv(bench_csv, index=False)
    print(f"\nSaved benchmark comparison to {bench_csv}")
    print(df_bench.to_string(index=False))

    if sample_visuals:
        fig, axes = plt.subplots(len(sample_visuals), 4, figsize=(16, 4 * len(sample_visuals)))
        if len(sample_visuals) == 1:
            axes = np.expand_dims(axes, 0)

        for i, s in enumerate(sample_visuals):
            ct_norm = np.clip((s["ct"] - (-100)) / 350.0, 0, 1)

            axes[i, 0].imshow(ct_norm, cmap="gray")
            axes[i, 0].set_title(f"{s['case']} (z={s['slice_idx']})\nCT Slice", fontsize=11)
            axes[i, 0].axis("off")

            gt_overlay = np.stack([ct_norm]*3, axis=-1)
            gt_overlay[s["gt_liver"] > 0] = gt_overlay[s["gt_liver"] > 0] * 0.6 + np.array([0, 0.8, 0]) * 0.4
            gt_overlay[s["gt_tumor"] > 0] = gt_overlay[s["gt_tumor"] > 0] * 0.3 + np.array([1, 0, 0]) * 0.7
            axes[i, 1].imshow(gt_overlay)
            axes[i, 1].set_title("Ground Truth\n(Green: Liver, Red: Tumor)", fontsize=11)
            axes[i, 1].axis("off")

            pred_overlay = np.stack([ct_norm]*3, axis=-1)
            pred_overlay[s["pred_liver"] > 0] = pred_overlay[s["pred_liver"] > 0] * 0.6 + np.array([0, 0.8, 0]) * 0.4
            pred_overlay[s["pred_tumor"] > 0] = pred_overlay[s["pred_tumor"] > 0] * 0.3 + np.array([1, 0.2, 0]) * 0.7
            axes[i, 2].imshow(pred_overlay)
            axes[i, 2].set_title(f"Synergy Ensemble\nLiv Dice: {s['dice_liv']:.1f}%, Tum: {s['dice_tum']:.1f}%", fontsize=11)
            axes[i, 2].axis("off")

            comp = np.zeros((*ct_norm.shape, 3))
            comp[..., 0] = s["pred_tumor"]
            comp[..., 1] = s["gt_tumor"]
            comp[..., 2] = s["gt_liver"] * 0.2
            axes[i, 3].imshow(comp)
            axes[i, 3].set_title("Tumor Alignment (Yellow = True Pos)", fontsize=11)
            axes[i, 3].axis("off")

        plt.tight_layout()
        vis_path = "results/ensemble_synergy_visual.png"
        plt.savefig(vis_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nSaved visualization panel to {vis_path}")

if __name__ == "__main__":
    main()
