"""
PyTorch Dataset for WeakMedSAM on LiMT.
Handles 2D CT slices with primary labels (Liver, Tumor) and
Sub-Class Exploration (SCE) pseudo-labels for K=4 sub-classes.
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

class LiMTWeakMedSAMDataset(Dataset):
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
        s_id = row["slice_id"]

        img_path = os.path.join(self.img_dir, f"{s_id}.png")
        seg_path = os.path.join(self.seg_dir, f"{s_id}.png")

        img = Image.open(img_path).convert("RGB")
        seg = Image.open(seg_path).convert("L")

        # Data augmentation for training
        if self.is_train:
            if random.random() > 0.5:
                img = TF.hflip(img)
                seg = TF.hflip(seg)
            if random.random() > 0.5:
                img = TF.vflip(img)
                seg = TF.vflip(seg)
            if random.random() > 0.3:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                seg = TF.rotate(seg, angle, interpolation=TF.InterpolationMode.NEAREST)

        img_t = TF.to_tensor(img) # [3, 256, 256] in [0, 1]
        # Standard SAM normalization: mean=[123.675, 116.28, 103.53]/255, std=[58.395, 57.12, 57.375]/255
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1) / 255.0
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1) / 255.0
        img_norm = (img_t - pixel_mean) / pixel_std

        seg_arr = np.array(seg, dtype=np.uint8)
        # Binary masks
        liver_mask = (seg_arr >= 1).astype(np.float32)
        tumor_mask = (seg_arr >= 2).astype(np.float32)
        seg_dual = torch.from_numpy(np.stack([liver_mask, tumor_mask], axis=0)) # [2, 256, 256]

        # Primary classification label: [has_liver, has_tumor]
        plab = torch.tensor([float(liver_mask.sum() > 0), float(tumor_mask.sum() > 0)], dtype=torch.float32)

        # Sub-class label for SCE: length = 2 * child_classes + 1
        num_child_logits = 2 * self.child_classes + 1
        clab = torch.zeros(num_child_logits, dtype=torch.float32)
        if s_id in self.clusters:
            c_info = self.clusters[s_id] # [sub_liver, sub_tumor]
            has_any = False
            if plab[0] > 0:
                sub_c0 = int(c_info[0])
                clab[sub_c0 + 1] = 1.0
                has_any = True
            if plab[1] > 0:
                sub_c1 = int(c_info[1])
                clab[sub_c1 + 1 + self.child_classes] = 1.0
                has_any = True
            if not has_any:
                clab[0] = 1.0
        else:
            clab[0] = 1.0

        return {
            "img": img_norm,
            "raw_img": img_t,
            "plab": plab,
            "clab": clab,
            "seg": seg_dual,
            "slice_id": s_id,
            "case_id": row["case_id"],
            "slice_idx": int(row["slice_idx"]),
            "total_slices": int(row["total_slices"])
        }
