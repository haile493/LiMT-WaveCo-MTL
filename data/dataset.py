import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import cv2

# Canonical mapping from raw ITK-SNAP label to contiguous class index (0-5)
RAW_TO_CONTIGUOUS = {
    0: 0,  # Background
    1: 1,  # Liver parenchyma
    2: 2,  # Hepatic Cyst (Class 1)
    4: 3,  # Cavernous Hemangioma (Class 2)
    3: 4,  # Hepatic Metastasis (Class 3)
    6: 5,  # Primary Liver Cancer / HCC (Class 4)
}

CONTIGUOUS_TO_NAME = {
    0: "Background",
    1: "Liver",
    2: "Cyst",
    3: "Hemangioma",
    4: "Metastasis",
    5: "HCC"
}

def map_raw_seg_to_contiguous(seg_array):
    """Maps raw segmentation labels to contiguous integer indices [0..5]."""
    out = np.zeros_like(seg_array, dtype=np.int64)
    for raw_val, cont_val in RAW_TO_CONTIGUOUS.items():
        out[seg_array == raw_val] = cont_val
    return out

def get_liver_bbox(seg_mask, margin=15):
    """
    Finds 3D bounding box around liver + lesions (label > 0).
    Adds safety margin in all 3 dimensions.
    """
    pos = np.where(seg_mask > 0)
    if len(pos[0]) == 0:
        return [0, seg_mask.shape[0], 0, seg_mask.shape[1], 0, seg_mask.shape[2]]
    
    min_z, max_z = pos[0].min(), pos[0].max()
    min_y, max_y = pos[1].min(), pos[1].max()
    min_x, max_x = pos[2].min(), pos[2].max()
    
    min_z = max(0, min_z - margin)
    max_z = min(seg_mask.shape[0], max_z + margin + 1)
    min_y = max(0, min_y - margin)
    max_y = min(seg_mask.shape[1], max_y + margin + 1)
    min_x = max(0, min_x - margin)
    max_x = min(seg_mask.shape[2], max_x + margin + 1)
    
    return [min_z, max_z, min_y, max_y, min_x, max_x]

def apply_clahe_3d(vol, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) slice-by-slice
    along the axial dimension for a normalized [0, 1] 3D CT volume.
    As demonstrated in the MSFF paper, CLAHE significantly sharpens lesion edges
    and boundary contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    out = np.zeros_like(vol, dtype=np.float32)
    for z in range(vol.shape[0]):
        slice_2d = vol[z]
        slice_uint8 = np.clip(slice_2d * 255.0, 0, 255).astype(np.uint8)
        slice_enhanced = clahe.apply(slice_uint8)
        out[z] = slice_enhanced.astype(np.float32) / 255.0
    return out

def resize_volume_and_mask(vol, mask, target_shape=(96, 96, 96)):
    """
    Resizes 3D volume (trilinear/spline) and 3D mask (nearest neighbor)
    to target shape.
    """
    factors = [target_shape[i] / vol.shape[i] for i in range(3)]
    vol_resized = ndimage.zoom(vol, factors, order=1, prefilter=False)
    mask_resized = ndimage.zoom(mask, factors, order=0, prefilter=False)
    return vol_resized, mask_resized

class LiMTMultiTaskDataset(Dataset):
    """
    Multi-task Dataset for LiMT:
    Provides paired (CT Volume, Segmentation Mask, Multi-label Classification Vector).
    Enhanced with CLAHE preprocessing and spatial augmentations.
    """
    def __init__(
        self,
        data_dir="LiMT Dataset",
        splits_csv="LiMT Dataset/limt_splits.csv",
        split_type="3way", # '3way' or 'kfold'
        split_val="train", # 'train', 'val', 'test' for 3way, or integer 0..4 for fold
        target_shape=(64, 64, 64),
        hu_min=-100.0,
        hu_max=250.0,
        use_clahe=True,
        augment=False,
        cache_data=False
    ):
        self.data_dir = data_dir
        self.ct_dir = os.path.join(data_dir, "ct")
        self.seg_dir = os.path.join(data_dir, "seg")
        self.target_shape = target_shape
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.use_clahe = use_clahe
        self.augment = augment
        self.cache_data = cache_data
        self.cache = {}
        
        df = pd.read_csv(splits_csv)
        if split_type == "3way":
            self.df = df[df['split_3way'] == split_val].reset_index(drop=True)
        elif split_type == "kfold":
            if isinstance(split_val, tuple) and split_val[0] == "val":
                fold_num = split_val[1]
                self.df = df[df['fold'] == fold_num].reset_index(drop=True)
            elif isinstance(split_val, tuple) and split_val[0] == "train":
                fold_num = split_val[1]
                self.df = df[df['fold'] != fold_num].reset_index(drop=True)
            else:
                self.df = df[df['fold'] == split_val].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)
            
        print(f"Loaded {len(self.df)} cases for split={split_val} ({split_type}), CLAHE={use_clahe}")

    def __len__(self):
        return len(self.df)

    def _load_case(self, idx):
        row = self.df.iloc[idx]
        ct_name = row['Filename']
        seg_name = ct_name.replace('volume_', 'segmentation_')
        
        ct_path = os.path.join(self.ct_dir, ct_name)
        seg_path = os.path.join(self.seg_dir, seg_name)
        
        ct_nib = nib.load(ct_path)
        seg_nib = nib.load(seg_path)
        
        ct_data = ct_nib.get_fdata().astype(np.float32)
        seg_data = seg_nib.get_fdata().astype(np.int64)
        
        # Orient to standard shape (Z, Y, X)
        ct_data = np.transpose(ct_data, (2, 1, 0))
        seg_data = np.transpose(seg_data, (2, 1, 0))
        
        # Map raw labels to contiguous [0..5]
        seg_mapped = map_raw_seg_to_contiguous(seg_data)
        
        # Liver Bounding-Box crop
        bbox = get_liver_bbox(seg_mapped, margin=15)
        min_z, max_z, min_y, max_y, min_x, max_x = bbox
        
        ct_crop = ct_data[min_z:max_z, min_y:max_y, min_x:max_x]
        seg_crop = seg_mapped[min_z:max_z, min_y:max_y, min_x:max_x]
        
        # Intensity windowing to [0, 1]
        ct_norm = np.clip(ct_crop, self.hu_min, self.hu_max)
        ct_norm = (ct_norm - self.hu_min) / (self.hu_max - self.hu_min)
        
        # Apply CLAHE enhancement
        if self.use_clahe:
            ct_norm = apply_clahe_3d(ct_norm)
            
        # Resize to target shape
        ct_res, seg_res = resize_volume_and_mask(ct_norm, seg_crop, self.target_shape)
        
        # Multi-label classification vector: [0, 1, 2, 3, 4]
        cls_labels = row[['0', '1', '2', '3', '4']].values.astype(np.float32)
        
        return ct_res, seg_res, cls_labels, ct_name

    def __getitem__(self, idx):
        if self.cache_data and idx in self.cache:
            ct_vol, seg_mask, cls_labels, ct_name = self.cache[idx]
        else:
            ct_vol, seg_mask, cls_labels, ct_name = self._load_case(idx)
            if self.cache_data:
                self.cache[idx] = (ct_vol, seg_mask, cls_labels, ct_name)
                
        # Data Augmentations
        if self.augment:
            if np.random.rand() > 0.5:
                ct_vol = np.flip(ct_vol, axis=1) # Y flip
                seg_mask = np.flip(seg_mask, axis=1)
            if np.random.rand() > 0.5:
                ct_vol = np.flip(ct_vol, axis=2) # X flip
                seg_mask = np.flip(seg_mask, axis=2)
            if np.random.rand() > 0.5:
                k = np.random.choice([1, 2, 3])
                ct_vol = np.rot90(ct_vol, k, axes=(1, 2))
                seg_mask = np.rot90(seg_mask, k, axes=(1, 2))

        # Add channel dimension: (1, D, H, W)
        ct_tensor = torch.from_numpy(np.ascontiguousarray(ct_vol, dtype=np.float32)).unsqueeze(0)
        seg_tensor = torch.from_numpy(np.ascontiguousarray(seg_mask, dtype=np.int64))
        cls_tensor = torch.from_numpy(np.ascontiguousarray(cls_labels, dtype=np.float32))

        # Tumor Burden Percentage (%): Ratio of tumor voxels to liver voxels
        tumor_voxels = float(np.sum(seg_mask >= 2))
        liver_voxels = float(np.sum(seg_mask >= 1))
        burden_val = (tumor_voxels / max(liver_voxels, 1.0)) * 100.0
        burden_tensor = torch.tensor([burden_val], dtype=torch.float32)

        return {
            'image': ct_tensor,
            'segmentation': seg_tensor,
            'classification': cls_tensor,
            'tumor_burden': burden_tensor,
            'filename': ct_name
        }

if __name__ == "__main__":
    ds = LiMTMultiTaskDataset(split_type="3way", split_val="val", target_shape=(64, 64, 64), use_clahe=True)
    sample = ds[0]
    print("Sample with CLAHE:", sample['filename'])
    print("Image shape:", sample['image'].shape, "min:", sample['image'].min(), "max:", sample['image'].max())
