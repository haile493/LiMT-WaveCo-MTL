"""
Evaluation script for SynPo on Few-Shot Tumor / Lesion Segmentation on LiMT.
Compares PerSAM baseline vs. SynPo (MICCAI 2025) on tumor-bearing test cases.
"""

import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
import torch
from tqdm import tqdm

from synpo_method import SynPoSegmenter, min_max_normalize, set_seed
from metrics import compute_binary_dice_jaccard


def preprocess_ct_slice(slice_2d, hu_min=-100.0, hu_max=250.0):
    clipped = np.clip(slice_2d, hu_min, hu_max)
    norm = ((clipped - hu_min) / (hu_max - hu_min) * 255.0).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


def persam_point_selection(sam_sim_map, Kp=3, Kn=3):
    H, W = sam_sim_map.shape
    pos_flat = np.argsort(sam_sim_map.ravel())[::-1][:Kp]
    py, px = np.unravel_index(pos_flat, (H, W))
    pos_pts = np.stack([px, py], axis=1)

    neg_flat = np.argsort(sam_sim_map.ravel())[:Kn]
    ny, nx = np.unravel_index(neg_flat, (H, W))
    neg_pts = np.stack([nx, ny], axis=1)

    coords = np.vstack([pos_pts, neg_pts])
    labels = np.array([1] * Kp + [0] * Kn, dtype=np.int32)
    return coords, labels


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Initializing SynPo Tumor Segmenter on {device} ===")
    segmenter = SynPoSegmenter(device=device, alpha=0.0, beta=-1.5, Kp=3, Kn=3)

    # 1. Build class-specific support representations
    # Class 1 (Cyst): volume_12_1
    # Class 2 (Hemangioma): volume_100_2
    # Class 3 (Metastasis): volume_15_3
    # Class 4 (HCC): volume_2_14
    support_cases = {
        1: "volume_12_1.nii",
        2: "volume_100_2.nii",
        3: "volume_15_3.nii",
        4: "volume_2_14.nii"
    }

    support_reprs = {}
    for cls_id, fname in support_cases.items():
        v_path = os.path.join("LiMT Dataset/ct", fname)
        s_path = os.path.join("LiMT Dataset/seg", fname.replace("volume_", "segmentation_"))
        vol = nib.load(v_path).get_fdata()
        seg = nib.load(s_path).get_fdata()

        # Target lesion mask (in LiMT raw format, class 1=2, 2=4, 3=3, 4=6, or all lesions >= 2)
        # Find slice with maximum tumor voxels
        t_counts = [(seg[:, :, z] >= 2).sum() for z in range(vol.shape[2])]
        best_z = int(np.argmax(t_counts))
        rgb = preprocess_ct_slice(vol[:, :, best_z])
        mask = (seg[:, :, best_z] >= 2).astype(np.uint8)

        print(f"Support for Class {cls_id}: {fname} (z={best_z}, tumor voxels={t_counts[best_z]})")
        support_reprs[cls_id] = segmenter.extract_support_representation(rgb, mask)

    # Also build a general tumor support prototype by averaging
    combined_repr = {
        "sam_fg_proto": torch.mean(torch.stack([r["sam_fg_proto"] for r in support_reprs.values()]), dim=0),
        "sam_bg_proto": torch.mean(torch.stack([r["sam_bg_proto"] for r in support_reprs.values()]), dim=0),
        "dino_fg_proto": torch.mean(torch.stack([r["dino_fg_proto"] for r in support_reprs.values()]), dim=0),
        "dino_bg_proto": torch.mean(torch.stack([r["dino_bg_proto"] for r in support_reprs.values()]), dim=0),
    }

    # 2. Test Cases (20 tumor-bearing cases)
    splits_df = pd.read_csv("LiMT Dataset/limt_splits.csv")
    test_df = splits_df[splits_df["split_3way"] == "test"]
    tumor_cases = test_df[(test_df["1"] == 1) | (test_df["2"] == 1) | (test_df["3"] == 1) | (test_df["4"] == 1)]

    persam_dices = []
    persam_jaccs = []
    persam_sens = []
    persam_precs = []

    synpo_dices = []
    synpo_jaccs = []
    synpo_sens = []
    synpo_precs = []

    case_records = []

    for _, row in tqdm(tumor_cases.iterrows(), total=len(tumor_cases), desc="Evaluating Tumor Cases"):
        fname = row["Filename"]
        v_path = os.path.join("LiMT Dataset/ct", fname)
        s_path = os.path.join("LiMT Dataset/seg", fname.replace("volume_", "segmentation_"))

        q_vol = nib.load(v_path).get_fdata()
        q_seg = nib.load(s_path).get_fdata()

        # Tumor slices (>= 50 tumor voxels)
        t_slices = [z for z in range(q_vol.shape[2]) if (q_seg[:, :, z] >= 2).sum() > 50]
        if len(t_slices) == 0:
            continue

        c_p_dice, c_p_jacc, c_p_sens, c_p_prec = [], [], [], []
        c_s_dice, c_s_jacc, c_s_sens, c_s_prec = [], [], [], []

        for qz in t_slices:
            gt_tumor = (q_seg[:, :, qz] >= 2).astype(bool)
            gt_liver = (q_seg[:, :, qz] >= 1).astype(bool)
            q_rgb = preprocess_ct_slice(q_vol[:, :, qz])

            H, W = q_rgb.shape[:2]
            sam_feat = segmenter.extract_sam_features(q_rgb)
            dino_feat = segmenter.extract_dino_features(q_rgb)

            # Cosine similarity maps
            s_sam = torch.sum(sam_feat * combined_repr["sam_fg_proto"].unsqueeze(-1), dim=0).cpu().numpy()
            s_sam_neg = torch.sum(sam_feat * combined_repr["sam_bg_proto"].unsqueeze(-1), dim=0).cpu().numpy()
            s_dino = torch.sum(dino_feat * combined_repr["dino_fg_proto"].unsqueeze(-1), dim=0).cpu().numpy()
            s_dino_neg = torch.sum(dino_feat * combined_repr["dino_bg_proto"].unsqueeze(-1), dim=0).cpu().numpy()

            s_sam_norm = min_max_normalize(s_sam)
            s_sam_neg_norm = min_max_normalize(s_sam_neg)
            s_dino_norm = min_max_normalize(s_dino)
            s_dino_neg_norm = min_max_normalize(s_dino_neg)

            # Synergy map
            syn_map = 0.8 * (s_sam_norm * s_dino_norm) + 0.1 * s_sam_norm + 0.1 * s_dino_norm
            syn_map_neg = 0.8 * (s_sam_neg_norm * s_dino_neg_norm) + 0.1 * s_sam_neg_norm + 0.1 * s_dino_neg_norm

            # PerSAM
            s_sam_up = cv2.resize(s_sam_norm, (W, H))
            persam_coords, persam_labels = persam_point_selection(s_sam_up, Kp=3, Kn=3)
            m_persam, sc_persam, _ = segmenter.predictor.predict(
                point_coords=persam_coords, point_labels=persam_labels, multimask_output=True
            )
            pred_persam = m_persam[np.argmax(sc_persam)] & gt_liver  # constrained to liver

            # SynPo
            syn_coords, syn_labels, _ = segmenter.point_selection_module(syn_map, syn_map_neg, (H, W))
            m_synpo, sc_synpo, _ = segmenter.predictor.predict(
                point_coords=syn_coords, point_labels=syn_labels, multimask_output=True
            )
            pred_synpo = m_synpo[np.argmax(sc_synpo)] & gt_liver  # constrained to liver

            # Metrics
            d_p, j_p = compute_binary_dice_jaccard(pred_persam, gt_tumor)
            inter_p = np.logical_and(pred_persam, gt_tumor).sum()
            sens_p = (inter_p / gt_tumor.sum() * 100.0) if gt_tumor.sum() > 0 else 0.0
            prec_p = (inter_p / pred_persam.sum() * 100.0) if pred_persam.sum() > 0 else 0.0

            d_s, j_s = compute_binary_dice_jaccard(pred_synpo, gt_tumor)
            inter_s = np.logical_and(pred_synpo, gt_tumor).sum()
            sens_s = (inter_s / gt_tumor.sum() * 100.0) if gt_tumor.sum() > 0 else 0.0
            prec_s = (inter_s / pred_synpo.sum() * 100.0) if pred_synpo.sum() > 0 else 0.0

            c_p_dice.append(d_p)
            c_p_jacc.append(j_p)
            c_p_sens.append(sens_p)
            c_p_prec.append(prec_p)

            c_s_dice.append(d_s)
            c_s_jacc.append(j_s)
            c_s_sens.append(sens_s)
            c_s_prec.append(prec_s)

        if len(c_p_dice) > 0:
            persam_dices.append(np.mean(c_p_dice))
            persam_jaccs.append(np.mean(c_p_jacc))
            persam_sens.append(np.mean(c_p_sens))
            persam_precs.append(np.mean(c_p_prec))

            synpo_dices.append(np.mean(c_s_dice))
            synpo_jaccs.append(np.mean(c_s_jacc))
            synpo_sens.append(np.mean(c_s_sens))
            synpo_precs.append(np.mean(c_s_prec))

            case_records.append({
                "Filename": fname,
                "PerSAM_Dice": np.mean(c_p_dice),
                "SynPo_Dice": np.mean(c_s_dice),
                "Gain": np.mean(c_s_dice) - np.mean(c_p_dice)
            })

    # Save Results
    res_df = pd.DataFrame([
        {
            "Method": "PerSAM (ICLR 2024)",
            "Tumor Dice (%)": f"{np.mean(persam_dices):.2f} ± {np.std(persam_dices):.2f}",
            "Tumor Jaccard (%)": f"{np.mean(persam_jaccs):.2f}",
            "Sensitivity (%)": f"{np.mean(persam_sens):.2f}",
            "Precision (%)": f"{np.mean(persam_precs):.2f}",
        },
        {
            "Method": "SynPo (MICCAI 2025)",
            "Tumor Dice (%)": f"{np.mean(synpo_dices):.2f} ± {np.std(synpo_dices):.2f}",
            "Tumor Jaccard (%)": f"{np.mean(synpo_jaccs):.2f}",
            "Sensitivity (%)": f"{np.mean(synpo_sens):.2f}",
            "Precision (%)": f"{np.mean(synpo_precs):.2f}",
        }
    ])

    res_df.to_csv("results/synpo_tumor_evaluation_limt.csv", index=False)
    pd.DataFrame(case_records).to_csv("results/synpo_tumor_per_case.csv", index=False)

    print("\n=== SUMMARY TABLE (LiMT Few-Shot Tumor Segmentation Benchmark) ===")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
