"""
Prompt Affinity Mining (PAM) and Random Walk Refinement for WeakMedSAM on LiMT.
Implements Section III-C of WeakMedSAM:
1. Grid Point Prompting on SAM prompt encoder + mask decoder
2. Global Affinity Map generation M_aff
3. Pairwise Structural Affinity Matrix A with power beta suppression
4. Random Walk diffusion vec(M*) = T^t * vec(CAM)
5. Refined pseudo-mask generation for liver and tumor
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import sys
import os

# WeakMedSAM imports
sys.path.append(os.path.abspath("ref/WeakMedSAM-main"))
from utils.torchutils import max_norm
from utils.affinity import get_indices_of_pairs

class PromptAffinityMiner:
    def __init__(self, model, beta=4, t=4, grid_ratio=4, radius=5, device="cuda"):
        self.model = model
        self.beta = beta
        self.t = t
        self.grid_ratio = grid_ratio
        self.radius = radius
        self.device = device

        # Precompute grid prompt coordinates (in 256x256 image coordinate space)
        coords = []
        step = 256 // grid_ratio
        for i in range(grid_ratio):
            for j in range(grid_ratio):
                coords.append([int((j + 0.5) * step), int((i + 0.5) * step)])
        self.num_pts = len(coords)
        self.prompt_coords = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(1) # [16, 1, 2]
        self.prompt_labels = torch.ones((self.num_pts, 1), dtype=torch.int32, device=device)

        # Precompute affinity index graph for 64x64 feature resolution
        self.ind_from, self.ind_to = get_indices_of_pairs(radius, (64, 64))
        self.ind_from = torch.from_numpy(self.ind_from).to(device)
        self.ind_to = torch.from_numpy(self.ind_to).to(device)

    @torch.no_grad()
    def compute_affinity_matrix(self, seg_embeddings):
        """
        Computes transition probability matrix T using batched prompt grid evaluation on SAM.
        seg_embeddings: [B, 256, 64, 64] image embeddings from SAM image encoder
        """
        B = seg_embeddings.shape[0]
        # Repeat embedding for all prompt points
        emb_repeat = seg_embeddings.repeat_interleave(self.num_pts, dim=0) # [B*16, 256, 64, 64]
        coords_batch = self.prompt_coords.repeat(B, 1, 1) # [B*16, 1, 2]
        labels_batch = self.prompt_labels.repeat(B, 1)

        se, de = self.model.prompt_encoder(points=(coords_batch, labels_batch), boxes=None, masks=None)
        low_res_masks, _ = self.model.mask_decoder(
            image_embeddings=emb_repeat,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=False
        )

        # Sum masks across the 16 grid prompts per sample: [B, 1, 128, 128]
        low_res_masks = low_res_masks.view(B, self.num_pts, 1, low_res_masks.shape[2], low_res_masks.shape[3])
        aff_map = low_res_masks.sum(dim=1) # [B, 1, 128, 128]
        aff_map_64 = F.interpolate(aff_map, (64, 64), mode="bilinear", align_corners=False) # [B, 1, 64, 64]
        aff_norm = max_norm(aff_map_64)

        from utils.affinity import get_aff
        aff_mat = torch.pow(get_aff(aff_norm), self.beta)
        T = aff_mat / (torch.sum(aff_mat, dim=1, keepdim=True) + 1e-8)
        return T, aff_map_64

    @torch.no_grad()
    def refine_cam(self, cam, seg_embeddings, pred_probs=None, threshold=0.5):
        """
        Refines raw CAM via PAM Random Walk.
        cam: [B, num_classes, 256, 256]
        seg_embeddings: [B, 256, 64, 64]
        pred_probs: [B, num_classes] (optional gating by slice classification)
        """
        B, C, H, W = cam.shape
        T, aff_map = self.compute_affinity_matrix(seg_embeddings)

        # Downsample CAM to 64x64 for random walk propagation
        rw_cam = F.interpolate(cam, (64, 64), mode="bilinear", align_corners=False) # [B, C, 64, 64]
        flat_cam = rw_cam.view(B, C, -1).permute(0, 2, 1) # [B, N, C]

        # Iterative Random Walk: vec(M*) = T^t * vec(CAM)
        for _ in range(self.t):
            flat_cam = torch.bmm(T, flat_cam)
            rw_cam = flat_cam.permute(0, 2, 1).view(B, C, 64, 64)
            rw_cam = max_norm(rw_cam)
            flat_cam = rw_cam.view(B, C, -1).permute(0, 2, 1)

        refined_cam = F.interpolate(rw_cam, (H, W), mode="bilinear", align_corners=False)
        # Apply slight spatial smoothing to boundary transitions
        refined_cam = TF.gaussian_blur(refined_cam, kernel_size=11, sigma=[1.5, 1.5])

        if pred_probs is not None:
            gate = pred_probs.view(B, C, 1, 1).expand_as(refined_cam)
            refined_cam = refined_cam * gate

        refined_masks = (refined_cam >= threshold).float()
        return refined_cam, refined_masks, aff_map
