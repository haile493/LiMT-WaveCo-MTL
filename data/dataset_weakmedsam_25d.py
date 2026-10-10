"""
2.5D Multi-Slice Dataset for WeakMedSAM on LiMT.
Loads 3 consecutive axial slices [z-1, z, z+1] as a 3-channel input
to provide through-plane 3D anatomical continuity while maintaining high in-plane resolution.
"""

import os
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import random

class LiMTWeakMedSAM25DDataset(Dataset):
    def __init__(
        self,
        data_dir="LiMT_WeakMedSAM",
        split="train",
        child_classes=4,
        cluster_file="LiMT_WeakMedSAM/limt_clusters_k4.pkl",
        is_train=True,
    ):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.seg_dir = os.path.join(data_dir, "masks")
        self.split = split
        self.is_train = is_train
        self.child_classes = child_classes

        meta_path = os.path.join(data_dir, "slices_meta.csv")
        df = pd.read_csv(meta_path)
        if split in ["train", "val", "test"]:
            self.df = df[df["split"] == split].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        if os.path.exists(cluster_file):
            with open(cluster_file, "rb") as f:
                self.clusters = pickle.load(f)
        else:
            self.clusters = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]
        z = int(row["slice_idx"])
        total_slices = int(row["total_slices"])

        # 2.5D slice indices: [z-1, z, z+1]
        z_prev = max(0, z - 1)
        z_next = min(total_slices - 1, z + 1)

        path_prev = os.path.join(self.img_dir, f"{case_id}_z{z_prev:03d}.png")
        path_curr = os.path.join(self.img_dir, f"{case_id}_z{z:03d}.png")
        path_next = os.path.join(self.img_dir, f"{case_id}_z{z_next:03d}.png")
        seg_path = os.path.join(self.seg_dir, f"{case_id}_z{z:03d}.png")

        img_prev = Image.open(path_prev).convert("L")
        img_curr = Image.open(path_curr).convert("L")
        img_next = Image.open(path_next).convert("L")
        seg = Image.open(seg_path).convert("L")

        # Data augmentation for training
        if self.is_train:
            if random.random() > 0.5:
                img_prev = TF.hflip(img_prev)
                img_curr = TF.hflip(img_curr)
                img_next = TF.hflip(img_next)
                seg = TF.hflip(seg)
            if random.random() > 0.5:
                img_prev = TF.vflip(img_prev)
                img_curr = TF.vflip(img_curr)
                img_next = TF.vflip(img_next)
                seg = TF.vflip(seg)
            if random.random() > 0.3:
                angle = random.uniform(-15, 15)
                img_prev = TF.rotate(img_prev, angle, interpolation=TF.InterpolationMode.BILINEAR)
                img_curr = TF.rotate(img_curr, angle, interpolation=TF.InterpolationMode.BILINEAR)
                img_next = TF.rotate(img_next, angle, interpolation=TF.InterpolationMode.BILINEAR)
                seg = TF.rotate(seg, angle, interpolation=TF.InterpolationMode.NEAREST)

        t_prev = TF.to_tensor(img_prev) # [1, 256, 256]
        t_curr = TF.to_tensor(img_curr)
        t_next = TF.to_tensor(img_next)
        img_25d = torch.cat([t_prev, t_curr, t_next], dim=0) # [3, 256, 256]

        # Normalization with ImageNet / SAM stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_norm = (img_25d - mean) / std

        seg_arr = np.array(seg, dtype=np.uint8)
        liver_mask = (seg_arr >= 1).astype(np.float32)
        tumor_mask = (seg_arr >= 2).astype(np.float32)
        seg_dual = torch.from_numpy(np.stack([liver_mask, tumor_mask], axis=0))

        plab = torch.tensor([float(liver_mask.sum() > 0), float(tumor_mask.sum() > 0)], dtype=torch.float32)

        s_id = f"{case_id}_z{z:03d}"
        num_child_logits = 2 * self.child_classes + 1
        clab = torch.zeros(num_child_logits, dtype=torch.float32)
        if s_id in self.clusters:
            c_info = self.clusters[s_id]
            has_any = False
            if plab[0] > 0:
                clab[int(c_info[0]) + 1] = 1.0
                has_any = True
            if plab[1] > 0:
                clab[int(c_info[1]) + 1 + self.child_classes] = 1.0
                has_any = True
            if not has_any:
                clab[0] = 1.0
        else:
            clab[0] = 1.0

        return {
            "img": img_norm,
            "raw_img": img_25d,
            "plab": plab,
            "clab": clab,
            "seg": seg_dual,
            "slice_id": s_id,
            "case_id": case_id,
            "slice_idx": z,
            "total_slices": total_slices
        }
