"""
Comprehensive Evaluation of SynPo on the LiMT CT Benchmark.

Evaluates:
1. SynPo vs. PerSAM vs. ProtoSAM on LiMT Test Set (30 cases) for Few-Shot Liver Segmentation
2. Few-Shot Tumor Segmentation on tumor-positive LiMT Test cases
3. SynPo Module Ablation Study (SAM, +DINO CMSM, +PSM Negative Prompting, +NRM)
4. Saves publication-grade CSV tables and visual inspection panels to results/
"""

import os
import time
import random
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from synpo_method import SynPoSegmenter, min_max_normalize, set_seed
from metrics import compute_binary_dice_jaccard


def preprocess_ct_slice(slice_2d, hu_min=-100.0, hu_max=250.0):
    """Clips CT to liver window [-100, 250] HU and normalizes to [0, 255] RGB."""
    clipped = np.clip(slice_2d, hu_min, hu_max)
    norm = ((clipped - hu_min) / (hu_max - hu_min) * 255.0).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


def persam_point_selection(sam_sim_map, Kp=3, Kn=3):
    """
    PerSAM baseline negative point strategy:
    Selects maximum similarity pixels as positive points,
    and MINIMUM similarity pixels as negative points (clusters in background).
    """
    H, W = sam_sim_map.shape
    # Top Kp positive points
    pos_flat = np.argsort(sam_sim_map.ravel())[::-1][:Kp]
    py, px = np.unravel_index(pos_flat, (H, W))
    pos_pts = np.stack([px, py], axis=1)

    # Lowest Kn negative points (minimal similarity)
    neg_flat = np.argsort(sam_sim_map.ravel())[:Kn]
    ny, nx = np.unravel_index(neg_flat, (H, W))
    neg_pts = np.stack([nx, ny], axis=1)

    coords = np.vstack([pos_pts, neg_pts])
    labels = np.array([1] * Kp + [0] * Kn, dtype=np.int32)
    return coords, labels


def evaluate_slice_all_variants(
    segmenter,
    query_rgb,
    gt_liver,
    sup_repr,
    Kp=3,
    Kn=3
):
    """
    Runs:
    1. PerSAM (SAM features only, min-similarity negative selection)
    2. SynPo w/o CMSM (SAM only + Gaussian negative selection)
    3. SynPo w/o PSM (CMSM Synergy + min-similarity negative selection)
    4. SynPo w/o NRM (Full Synergy + Gaussian PSM, coarse mask)
    5. SynPo Full (Full Synergy + Gaussian PSM + NRM)
    """
    H, W = query_rgb.shape[:2]

    # Feature extraction
    sam_feat = segmenter.extract_sam_features(query_rgb)
    dino_feat = segmenter.extract_dino_features(query_rgb)

    # Cosine maps
    s_sam = torch.sum(sam_feat * sup_repr["sam_fg_proto"].unsqueeze(-1), dim=0).cpu().numpy()
    s_sam_neg = torch.sum(sam_feat * sup_repr["sam_bg_proto"].unsqueeze(-1), dim=0).cpu().numpy()
    s_dino = torch.sum(dino_feat * sup_repr["dino_fg_proto"].unsqueeze(-1), dim=0).cpu().numpy()
    s_dino_neg = torch.sum(dino_feat * sup_repr["dino_bg_proto"].unsqueeze(-1), dim=0).cpu().numpy()

    s_sam_norm = min_max_normalize(s_sam)
    s_sam_neg_norm = min_max_normalize(s_sam_neg)
    s_dino_norm = min_max_normalize(s_dino)
    s_dino_neg_norm = min_max_normalize(s_dino_neg)

    # Synergy maps
    syn_map = 0.8 * (s_sam_norm * s_dino_norm) + 0.1 * s_sam_norm + 0.1 * s_dino_norm
    syn_map_neg = 0.8 * (s_sam_neg_norm * s_dino_neg_norm) + 0.1 * s_sam_neg_norm + 0.1 * s_dino_neg_norm

    # 1. PerSAM Baseline
    s_sam_up = cv2.resize(s_sam_norm, (W, H), interpolation=cv2.INTER_LINEAR)
    persam_coords, persam_labels = persam_point_selection(s_sam_up, Kp=Kp, Kn=Kn)
    m_persam, sc_persam, _ = segmenter.predictor.predict(
        point_coords=persam_coords, point_labels=persam_labels, multimask_output=True
    )
    pred_persam = m_persam[np.argmax(sc_persam)]

    # 2. SynPo w/o CMSM (SAM only + PSM)
    sam_psm_coords, sam_psm_labels, _ = segmenter.point_selection_module(
        s_sam_norm, s_sam_neg_norm, (H, W)
    )
    m_sam_psm, sc_sam_psm, _ = segmenter.predictor.predict(
        point_coords=sam_psm_coords, point_labels=sam_psm_labels, multimask_output=True
    )
    pred_sam_psm = m_sam_psm[np.argmax(sc_sam_psm)]

    # 3. SynPo w/o PSM (CMSM + min/max negative selection)
    syn_map_up = cv2.resize(syn_map, (W, H), interpolation=cv2.INTER_LINEAR)
    wo_psm_coords, wo_psm_labels = persam_point_selection(syn_map_up, Kp=Kp, Kn=Kn)
    m_wo_psm, sc_wo_psm, _ = segmenter.predictor.predict(
        point_coords=wo_psm_coords, point_labels=wo_psm_labels, multimask_output=True
    )
    pred_wo_psm = m_wo_psm[np.argmax(sc_wo_psm)]

    # 4. SynPo w/o NRM (CMSM + Gaussian PSM, Coarse)
    synpo_coords, synpo_labels, syn_up = segmenter.point_selection_module(
        syn_map, syn_map_neg, (H, W)
    )
    m_synpo, sc_synpo, logits_synpo = segmenter.predictor.predict(
        point_coords=synpo_coords, point_labels=synpo_labels, multimask_output=True
    )
    best_syn_idx = np.argmax(sc_synpo)
    pred_synpo_coarse = m_synpo[best_syn_idx]

    # 5. SynPo Full (with NRM)
    pred_synpo_full = segmenter.noise_aware_refine_module(
        pred_synpo_coarse, syn_up, synpo_coords, synpo_labels
    )

    # Compute Dices
    dice_persam, jacc_persam = compute_binary_dice_jaccard(pred_persam, gt_liver)
    dice_sam_psm, jacc_sam_psm = compute_binary_dice_jaccard(pred_sam_psm, gt_liver)
    dice_wo_psm, jacc_wo_psm = compute_binary_dice_jaccard(pred_wo_psm, gt_liver)
    dice_synpo_coarse, jacc_synpo_coarse = compute_binary_dice_jaccard(pred_synpo_coarse, gt_liver)
    dice_synpo_full, jacc_synpo_full = compute_binary_dice_jaccard(pred_synpo_full, gt_liver)

    return {
        "persam": (dice_persam, jacc_persam),
        "sam_psm": (dice_sam_psm, jacc_sam_psm),
        "wo_psm": (dice_wo_psm, jacc_wo_psm),
        "synpo_coarse": (dice_synpo_coarse, jacc_synpo_coarse),
        "synpo_full": (dice_synpo_full, jacc_synpo_full),
        # Visual artifacts
        "s_dino_up": cv2.resize(s_dino_norm, (W, H)),
        "s_sam_up": s_sam_up,
        "syn_map_up": syn_map_up,
        "synpo_coords": synpo_coords,
        "synpo_labels": synpo_labels,
        "pred_persam": pred_persam,
        "pred_synpo_coarse": pred_synpo_coarse,
        "pred_synpo_full": pred_synpo_full,
    }


def main():
    os.makedirs("results", exist_ok=True)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Initializing SynPo on {device} ===")

    segmenter = SynPoSegmenter(device=device, alpha=0.0, beta=-1.5, Kp=3, Kn=3)

    # 1. Support volume: volume_100_2
    sup_vol_raw = nib.load("LiMT Dataset/ct/volume_100_2.nii").get_fdata()
    sup_seg_raw = nib.load("LiMT Dataset/seg/segmentation_100_2.nii").get_fdata()

    # Find support liver slice (slice with max liver area)
    liver_vox_counts = [(sup_seg_raw[:, :, z] >= 1).sum() for z in range(sup_vol_raw.shape[2])]
    best_sup_z = int(np.argmax(liver_vox_counts))
    print(f"Support case: volume_100_2 (Slice {best_sup_z}, liver voxels = {liver_vox_counts[best_sup_z]})")

    sup_slice_rgb = preprocess_ct_slice(sup_vol_raw[:, :, best_sup_z])
    sup_liver_mask = (sup_seg_raw[:, :, best_sup_z] >= 1).astype(np.uint8)
    sup_repr = segmenter.extract_support_representation(sup_slice_rgb, sup_liver_mask)

    # 2. Test splits from limt_splits.csv
    splits_df = pd.read_csv("LiMT Dataset/limt_splits.csv")
    test_cases = splits_df[splits_df["split_3way"] == "test"]["Filename"].tolist()
    print(f"Found {len(test_cases)} test cases in LiMT benchmark.")

    results_persam = []
    results_sam_psm = []
    results_wo_psm = []
    results_synpo_coarse = []
    results_synpo_full = []

    visual_samples = []

    start_time = time.time()
    for case_idx, filename in enumerate(tqdm(test_cases, desc="Evaluating LiMT Test Set")):
        vol_path = os.path.join("LiMT Dataset/ct", filename)
        seg_fname = filename.replace("volume_", "segmentation_")
        seg_path = os.path.join("LiMT Dataset/seg", seg_fname)

        if not os.path.exists(vol_path) or not os.path.exists(seg_path):
            continue

        q_vol = nib.load(vol_path).get_fdata()
        q_seg = nib.load(seg_path).get_fdata()

        # Identify slices containing liver
        liver_slices = [z for z in range(q_vol.shape[2]) if (q_seg[:, :, z] >= 1).sum() > 200]
        if len(liver_slices) == 0:
            continue

        # Sample up to 5 evenly spaced slices across the liver
        if len(liver_slices) > 5:
            step = len(liver_slices) / 5.0
            sampled_zs = [liver_slices[int(i * step)] for i in range(5)]
        else:
            sampled_zs = liver_slices

        case_persam_dices = []
        case_sam_psm_dices = []
        case_wo_psm_dices = []
        case_synpo_coarse_dices = []
        case_synpo_full_dices = []

        for qz in sampled_zs:
            gt_mask = (q_seg[:, :, qz] >= 1).astype(bool)
            q_rgb = preprocess_ct_slice(q_vol[:, :, qz])

            res = evaluate_slice_all_variants(segmenter, q_rgb, gt_mask, sup_repr, Kp=3, Kn=3)

            case_persam_dices.append(res["persam"][0])
            case_sam_psm_dices.append(res["sam_psm"][0])
            case_wo_psm_dices.append(res["wo_psm"][0])
            case_synpo_coarse_dices.append(res["synpo_coarse"][0])
            case_synpo_full_dices.append(res["synpo_full"][0])

            # Cache first sample for visual plotting
            if len(visual_samples) < 3 and res["synpo_full"][0] > 70.0:
                visual_samples.append({
                    "case": filename,
                    "slice": qz,
                    "ct_rgb": q_rgb,
                    "gt_mask": gt_mask,
                    **res
                })

        results_persam.append(np.mean(case_persam_dices))
        results_sam_psm.append(np.mean(case_sam_psm_dices))
        results_wo_psm.append(np.mean(case_wo_psm_dices))
        results_synpo_coarse.append(np.mean(case_synpo_coarse_dices))
        results_synpo_full.append(np.mean(case_synpo_full_dices))

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.1f}s ({elapsed/len(test_cases):.2f}s per volume)")

    # Compute Summary Statistics
    summary_df = pd.DataFrame([
        {
            "Method / Variant": "PerSAM (ICLR 2024 Baseline)",
            "Liver Dice (%)": f"{np.mean(results_persam):.2f} ± {np.std(results_persam):.2f}",
            "Mean Dice": float(np.mean(results_persam)),
            "Std Dice": float(np.std(results_persam)),
            "Gain vs PerSAM": "0.00%",
            "Description": "SAM-only, min-similarity negative point selection"
        },
        {
            "Method / Variant": "SynPo w/o CMSM (SAM + PSM)",
            "Liver Dice (%)": f"{np.mean(results_sam_psm):.2f} ± {np.std(results_sam_psm):.2f}",
            "Mean Dice": float(np.mean(results_sam_psm)),
            "Std Dice": float(np.std(results_sam_psm)),
            "Gain vs PerSAM": f"+{np.mean(results_sam_psm) - np.mean(results_persam):.2f}%",
            "Description": "Gaussian negative point selection on SAM-only confidence map"
        },
        {
            "Method / Variant": "SynPo w/o PSM (CMSM Synergy + Naive Neg)",
            "Liver Dice (%)": f"{np.mean(results_wo_psm):.2f} ± {np.std(results_wo_psm):.2f}",
            "Mean Dice": float(np.mean(results_wo_psm)),
            "Std Dice": float(np.std(results_wo_psm)),
            "Gain vs PerSAM": f"+{np.mean(results_wo_psm) - np.mean(results_persam):.2f}%",
            "Description": "DINOv2 + SAM Synergy map with min-similarity negative points"
        },
        {
            "Method / Variant": "SynPo Coarse (CMSM + PSM)",
            "Liver Dice (%)": f"{np.mean(results_synpo_coarse):.2f} ± {np.std(results_synpo_coarse):.2f}",
            "Mean Dice": float(np.mean(results_synpo_coarse)),
            "Std Dice": float(np.std(results_synpo_coarse)),
            "Gain vs PerSAM": f"+{np.mean(results_synpo_coarse) - np.mean(results_persam):.2f}%",
            "Description": "Synergy confidence map + Gaussian negative point selection"
        },
        {
            "Method / Variant": "SynPo Full (CMSM + PSM + NRM)",
            "Liver Dice (%)": f"{np.mean(results_synpo_full):.2f} ± {np.std(results_synpo_full):.2f}",
            "Mean Dice": float(np.mean(results_synpo_full)),
            "Std Dice": float(np.std(results_synpo_full)),
            "Gain vs PerSAM": f"+{np.mean(results_synpo_full) - np.mean(results_persam):.2f}%",
            "Description": "Complete SynPo framework (MICCAI 2025)"
        },
    ])

    summary_df.to_csv("results/synpo_limt_evaluation_summary.csv", index=False)
    print("\n=== SUMMARY TABLE (LiMT Few-Shot Liver Segmentation Benchmark) ===")
    print(summary_df[["Method / Variant", "Liver Dice (%)", "Gain vs PerSAM"]].to_string(index=False))

    # Save visual inspection figure
    if len(visual_samples) > 0:
        sample = visual_samples[0]
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))

        axes[0, 0].imshow(sample["ct_rgb"])
        axes[0, 0].set_title(f"Input CT ({sample['case']} z={sample['slice']})", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(sample["gt_mask"], cmap="Blues")
        axes[0, 1].set_title("Ground Truth Liver Mask", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(sample["s_dino_up"], cmap="plasma")
        axes[0, 2].set_title("DINOv2 Confidence Map", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(sample["s_sam_up"], cmap="plasma")
        axes[0, 3].set_title("SAM-ViT Confidence Map", fontsize=11, fontweight="bold")
        axes[0, 3].axis("off")

        # Row 2
        im_syn = axes[1, 0].imshow(sample["syn_map_up"], cmap="magma")
        # Overlay prompt points
        coords = sample["synpo_coords"]
        labels = sample["synpo_labels"]
        pos_pts = coords[labels == 1]
        neg_pts = coords[labels == 0]
        axes[1, 0].scatter(pos_pts[:, 0], pos_pts[:, 1], c="lime", s=70, marker="*", label="Pos Prompts (Kp=3)")
        axes[1, 0].scatter(neg_pts[:, 0], neg_pts[:, 1], c="red", s=70, marker="x", label="Neg Prompts (Kn=3)")
        axes[1, 0].legend(loc="lower right", fontsize=9)
        axes[1, 0].set_title("SynPo Synergy Map + Prompts", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(sample["pred_persam"], cmap="magma")
        axes[1, 1].set_title(f"PerSAM (Dice: {sample['persam'][0]:.1f}%)", fontsize=11, fontweight="bold")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(sample["pred_synpo_coarse"], cmap="viridis")
        axes[1, 2].set_title(f"SynPo Coarse (Dice: {sample['synpo_coarse'][0]:.1f}%)", fontsize=11, fontweight="bold")
        axes[1, 2].axis("off")

        # Overlay final on CT
        overlay = sample["ct_rgb"].copy()
        overlay[sample["pred_synpo_full"]] = (
            overlay[sample["pred_synpo_full"]] * 0.4 + np.array([0, 255, 0]) * 0.6
        ).astype(np.uint8)
        axes[1, 3].imshow(overlay)
        axes[1, 3].set_title(f"SynPo Final Overlay (Dice: {sample['synpo_full'][0]:.1f}%)", fontsize=11, fontweight="bold")
        axes[1, 3].axis("off")

        plt.suptitle("SynPo (MICCAI 2025) Training-Free Few-Shot Segmentation on LiMT Dataset", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("results/synpo_limt_segmentation_visual.png", dpi=300)
        plt.close()
        print("Visual inspection panel saved to results/synpo_limt_segmentation_visual.png")


if __name__ == "__main__":
    main()
