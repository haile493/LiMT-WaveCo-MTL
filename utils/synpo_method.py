"""
SynPo: Boosting Training-Free Few-Shot Medical Segmentation via High-Quality Negative Prompts (MICCAI 2025)
Implementation tailored for the LiMT (Liver Multi-Task) CT Dataset.

Components:
1. Feature Extraction: SAM-ViT (absolute positional cues) + DINOv2 (semantic features)
2. Confidence Map Synergy Module (CMSM): Eq. 1 - Eq. 6
3. Point Selection Module (PSM): Eq. 7 - Eq. 12
4. Noise-Aware Refine Module (NRM): Eq. 13 - Eq. 14
5. Full Training-Free Few-Shot Segmentation Pipeline for 2D slices and 3D LiMT CT volumes
"""

import os
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy import ndimage
from segment_anything import sam_model_registry, SamPredictor


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def min_max_normalize(feat_map, eps=1e-8):
    """Normalizes a 2D map to [0, 1]."""
    f_min = feat_map.min()
    f_max = feat_map.max()
    if f_max - f_min < eps:
        return np.zeros_like(feat_map)
    return (feat_map - f_min) / (f_max - f_min)


class SynPoSegmenter:
    """
    SynPo: Training-Free Few-Shot Medical Image Segmentation.
    """
    def __init__(
        self,
        sam_checkpoint="C:/Users/hai49/.gemini/antigravity/scratch/cangio/results/models/sam_vit_b_01ec64.pth",
        model_type="vit_b",
        dino_name="dinov2_vits14",
        device="cuda" if torch.cuda.is_available() else "cpu",
        delta_sd=0.8,
        delta_s=0.1,
        delta_d=0.1,
        alpha=0.0,
        beta=-1.5,
        gamma1=10,
        gamma2=20,
        Kp=3,
        Kn=3
    ):
        self.device = device
        self.delta_sd = delta_sd
        self.delta_s = delta_s
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.Kp = Kp
        self.Kn = Kn

        print(f"[SynPo] Initializing SAM ({model_type}) from {sam_checkpoint} on {device}...")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)

        print(f"[SynPo] Initializing DINOv2 ({dino_name}) on {device}...")
        self.dino = torch.hub.load("facebookresearch/dinov2", dino_name)
        self.dino.to(device=self.device)
        self.dino.eval()
        print("[SynPo] Model initialization complete.")

    @torch.no_grad()
    def extract_dino_features(self, img_rgb):
        """
        Extracts 2D DINOv2 patch features resized to spatial 64x64.
        Input: img_rgb (H, W, 3) uint8 or float [0, 255]
        Output: dino_feat (C_dino, 64, 64) torch tensor on cpu or device
        """
        # Resize to 896x896 so 896/14 = 64 exactly
        img_resized = cv2.resize(img_rgb, (896, 896), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        # ImageNet normalization for DINOv2
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.to(self.device)

        tokens = self.dino.get_intermediate_layers(tensor, n=1)[0]  # (1, 4096, C)
        # Reshape to (1, 64, 64, C) -> (1, C, 64, 64)
        c_dim = tokens.shape[-1]
        feat_2d = tokens.reshape(1, 64, 64, c_dim).permute(0, 3, 1, 2)
        # L2 normalize feature vectors along channel dimension
        feat_2d = F.normalize(feat_2d, p=2, dim=1)
        return feat_2d.squeeze(0)  # (C, 64, 64)

    @torch.no_grad()
    def extract_sam_features(self, img_rgb):
        """
        Extracts 2D SAM image encoder features of spatial shape (256, 64, 64).
        Also caches the image embedding in predictor for prompt inference.
        """
        self.predictor.set_image(img_rgb)
        sam_feat = self.predictor.features  # (1, 256, 64, 64)
        sam_feat = F.normalize(sam_feat, p=2, dim=1)
        return sam_feat.squeeze(0)  # (256, 64, 64)

    def extract_support_representation(self, support_img_rgb, support_mask_2d):
        """
        Extracts foreground and background prototype representations from support image.
        support_mask_2d: binary mask (H, W) where 1 is target organ/lesion.
        """
        # SAM features
        sam_feat = self.extract_sam_features(support_img_rgb)  # (256, 64, 64)
        # DINO features
        dino_feat = self.extract_dino_features(support_img_rgb)  # (384, 64, 64)

        # Downsample support mask to 64x64
        mask_64 = cv2.resize(support_mask_2d.astype(np.float32), (64, 64), interpolation=cv2.INTER_NEAREST)
        fg_mask = torch.from_numpy(mask_64 > 0.5).to(self.device)
        bg_mask = ~fg_mask

        # SAM Prototypes
        if fg_mask.sum() > 0:
            sam_fg_proto = sam_feat[:, fg_mask].mean(dim=-1, keepdim=True)  # (256, 1)
            sam_fg_proto = F.normalize(sam_fg_proto, p=2, dim=0)
        else:
            sam_fg_proto = torch.zeros((sam_feat.shape[0], 1), device=self.device)

        if bg_mask.sum() > 0:
            sam_bg_proto = sam_feat[:, bg_mask].mean(dim=-1, keepdim=True)
            sam_bg_proto = F.normalize(sam_bg_proto, p=2, dim=0)
        else:
            sam_bg_proto = torch.zeros((sam_feat.shape[0], 1), device=self.device)

        # DINO Prototypes
        if fg_mask.sum() > 0:
            dino_fg_proto = dino_feat[:, fg_mask].mean(dim=-1, keepdim=True)  # (384, 1)
            dino_fg_proto = F.normalize(dino_fg_proto, p=2, dim=0)
        else:
            dino_fg_proto = torch.zeros((dino_feat.shape[0], 1), device=self.device)

        if bg_mask.sum() > 0:
            dino_bg_proto = dino_feat[:, bg_mask].mean(dim=-1, keepdim=True)
            dino_bg_proto = F.normalize(dino_bg_proto, p=2, dim=0)
        else:
            dino_bg_proto = torch.zeros((dino_feat.shape[0], 1), device=self.device)

        return {
            "sam_fg_proto": sam_fg_proto,
            "sam_bg_proto": sam_bg_proto,
            "dino_fg_proto": dino_fg_proto,
            "dino_bg_proto": dino_bg_proto,
        }

    def compute_confidence_maps(self, query_sam_feat, query_dino_feat, support_repr):
        """
        Confidence Map Synergy Module (CMSM):
        Computes positive and negative confidence maps and fuses SAM + DINOv2.
        """
        # Cosine similarity maps: (64, 64)
        # S_SAM = sum_c (F_Q * T_S)
        s_sam = torch.sum(query_sam_feat * support_repr["sam_fg_proto"].unsqueeze(-1), dim=0).cpu().numpy()
        s_sam_neg = torch.sum(query_sam_feat * support_repr["sam_bg_proto"].unsqueeze(-1), dim=0).cpu().numpy()

        s_dino = torch.sum(query_dino_feat * support_repr["dino_fg_proto"].unsqueeze(-1), dim=0).cpu().numpy()
        s_dino_neg = torch.sum(query_dino_feat * support_repr["dino_bg_proto"].unsqueeze(-1), dim=0).cpu().numpy()

        # Min-max normalization of individual confidence maps
        s_sam = min_max_normalize(s_sam)
        s_sam_neg = min_max_normalize(s_sam_neg)
        s_dino = min_max_normalize(s_dino)
        s_dino_neg = min_max_normalize(s_dino_neg)

        # Synergy Map Fusion (Eq. 4 & Eq. 5)
        # SynMap = delta_sd * (S_SAM * S_DINO) + delta_s * S_SAM + delta_d * S_DINO
        syn_map = (
            self.delta_sd * (s_sam * s_dino)
            + self.delta_s * s_sam
            + self.delta_d * s_dino
        )
        syn_map_neg = (
            self.delta_sd * (s_sam_neg * s_dino_neg)
            + self.delta_s * s_sam_neg
            + self.delta_d * s_dino_neg
        )

        return syn_map, syn_map_neg

    def point_selection_module(self, syn_map, syn_map_neg, img_shape):
        """
        Point Selection Module (PSM):
        Selects Kp positive centroids and Kn negative centroids.
        img_shape: (H, W)
        """
        H, W = img_shape
        # Upsample synergy maps to target image resolution
        syn_map_up = cv2.resize(syn_map, (W, H), interpolation=cv2.INTER_LINEAR)
        syn_map_neg_up = cv2.resize(syn_map_neg, (W, H), interpolation=cv2.INTER_LINEAR)

        # 1. Negative Confidence Distribution Modeling (Gaussian fitting: Eq. 6)
        mu = np.mean(syn_map_neg_up)
        sigma = np.std(syn_map_neg_up) + 1e-8

        # 2. Positive Points Selection (Eq. 7 & Eq. 8)
        # Sort all pixels by confidence in descending order
        flat_indices = np.argsort(syn_map_up.ravel())[::-1]
        top_k_pos = max(self.Kp, int(self.gamma1 * self.Kp))
        top_pos_indices = flat_indices[:top_k_pos]

        pos_coords_y, pos_coords_x = np.unravel_index(top_pos_indices, (H, W))
        pos_points = np.stack([pos_coords_x, pos_coords_y], axis=1)  # (x, y) coordinates for SAM

        if len(pos_points) >= self.Kp:
            kmeans_pos = KMeans(n_clusters=self.Kp, random_state=42, n_init=10).fit(pos_points)
            pos_centroids = kmeans_pos.cluster_centers_  # (Kp, 2)
        else:
            pos_centroids = pos_points

        # 3. Negative Points Selection (Eq. 9 - Eq. 11)
        # Negative bounds: [mu - alpha*sigma, mu - beta*sigma]
        # In paper: Fig 4 shows e.g. [mu - sigma, mu + 0.5*sigma] or [mu, mu + 1.5*sigma]
        lower_bound = mu - self.alpha * sigma
        upper_bound = mu - self.beta * sigma
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        neg_mask = (syn_map_up >= lower_bound) & (syn_map_up <= upper_bound)
        neg_candidates_y, neg_candidates_x = np.where(neg_mask)

        # Fallback if candidate region has too few pixels
        if len(neg_candidates_x) < self.Kn:
            # Fallback to pixels with confidence below median but above minimum background
            med = np.median(syn_map_up)
            fallback_mask = (syn_map_up <= med) & (syn_map_up >= np.percentile(syn_map_up, 10))
            neg_candidates_y, neg_candidates_x = np.where(fallback_mask)

        neg_points_all = np.stack([neg_candidates_x, neg_candidates_y], axis=1)

        max_neg_sample = max(self.Kn, int(self.gamma2 * self.Kn))
        if len(neg_points_all) > max_neg_sample:
            idx = np.random.choice(len(neg_points_all), max_neg_sample, replace=False)
            neg_points_sample = neg_points_all[idx]
        else:
            neg_points_sample = neg_points_all

        if len(neg_points_sample) >= self.Kn:
            kmeans_neg = KMeans(n_clusters=self.Kn, random_state=42, n_init=10).fit(neg_points_sample)
            neg_centroids = kmeans_neg.cluster_centers_  # (Kn, 2)
        elif len(neg_points_sample) > 0:
            neg_centroids = neg_points_sample[:self.Kn]
        else:
            # Extreme fallback: image corners
            neg_centroids = np.array([[5, 5], [W - 5, 5], [W - 5, H - 5]], dtype=float)

        # Assemble prompt points and labels (Eq. 12)
        prompt_coords = np.vstack([pos_centroids, neg_centroids])
        prompt_labels = np.array([1] * len(pos_centroids) + [0] * len(neg_centroids), dtype=np.int32)

        return prompt_coords, prompt_labels, syn_map_up

    def noise_aware_refine_module(self, coarse_mask, syn_map_up, prompt_coords, prompt_labels):
        """
        Noise-Aware Refine Module (NRM):
        Eq. 13 - Eq. 14: Morphological opening, connected component confidence scoring,
        and SAM re-prediction with mask prompt.
        """
        # Morphological opening (erosion followed by dilation)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened_mask = cv2.morphologyEx(coarse_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # Connected components
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(opened_mask)

        if num_labels <= 1:
            # No component survived opening: fallback to coarse mask
            refined_candidate_mask = coarse_mask
        else:
            # Score each connected component (Eq. 14)
            best_score = -float("inf")
            best_component_idx = 1
            for j in range(1, num_labels):
                area = stats[j, cv2.CC_STAT_AREA]
                if area < 10:  # Skip tiny noise
                    continue
                comp_mask = (labels_im == j)
                score_j = float(np.mean(syn_map_up[comp_mask]))
                if score_j > best_score:
                    best_score = score_j
                    best_component_idx = j

            refined_candidate_mask = (labels_im == best_component_idx).astype(np.float32)

        # Feed refined candidate as mask prompt into SAM for fine-grained re-prediction
        # Low-res mask input expected by SAM is (1, 256, 256) logits
        mask_input_256 = cv2.resize(refined_candidate_mask, (256, 256), interpolation=cv2.INTER_LINEAR)
        # Convert binary [0, 1] to pseudo-logits: +10.0 for foreground, -10.0 for background
        mask_logits = (mask_input_256 - 0.5) * 20.0
        mask_input = mask_logits[None, :, :]  # (1, 256, 256)

        try:
            refined_masks, refined_scores, _ = self.predictor.predict(
                point_coords=prompt_coords,
                point_labels=prompt_labels,
                mask_input=mask_input,
                multimask_output=False,
            )
            final_mask = refined_masks[0]
        except Exception as e:
            print(f"[NRM Warning] Re-prediction fallback due to error: {e}")
            final_mask = refined_candidate_mask > 0.5

        return final_mask

    def segment_slice(self, query_img_rgb, support_repr, apply_nrm=True):
        """
        Executes full SynPo pipeline on a single 2D slice:
        1. Feature extraction for query slice (SAM + DINOv2)
        2. CMSM confidence map synergy
        3. PSM point selection
        4. Coarse SAM prediction
        5. NRM noise-aware refinement
        """
        H, W = query_img_rgb.shape[:2]

        # 1. Features
        sam_feat = self.extract_sam_features(query_img_rgb)
        dino_feat = self.extract_dino_features(query_img_rgb)

        # 2. CMSM
        syn_map, syn_map_neg = self.compute_confidence_maps(sam_feat, dino_feat, support_repr)

        # 3. PSM
        prompt_coords, prompt_labels, syn_map_up = self.point_selection_module(
            syn_map, syn_map_neg, (H, W)
        )

        # 4. SAM Coarse Prediction
        masks, scores, _ = self.predictor.predict(
            point_coords=prompt_coords,
            point_labels=prompt_labels,
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        coarse_mask = masks[best_idx]

        # 5. NRM
        if apply_nrm:
            final_mask = self.noise_aware_refine_module(
                coarse_mask, syn_map_up, prompt_coords, prompt_labels
            )
        else:
            final_mask = coarse_mask

        return {
            "coarse_mask": coarse_mask,
            "final_mask": final_mask,
            "prompt_coords": prompt_coords,
            "prompt_labels": prompt_labels,
            "syn_map": syn_map_up,
        }
