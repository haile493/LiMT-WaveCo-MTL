import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from dataset import LiMTMultiTaskDataset
from models.multitask_msff_net import MultiTaskLiverNet_MSFF
from metrics import compute_binary_dice_jaccard

def eval_thresholds(ckpt_path="checkpoints/best_msff_multitask_3way.pth", target_size=80):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskLiverNet_MSFF(in_channels=1, num_seg_classes=6, num_cls_classes=5, base_channels=16)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    test_ds = LiMTMultiTaskDataset(
        data_dir="LiMT Dataset", splits_csv="LiMT Dataset/limt_splits.csv",
        split_type="3way", split_val="test",
        target_shape=(target_size, target_size, target_size), use_clahe=True,
        augment=False, cache_data=True
    )
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    thresholds = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    res_per_thresh = {th: [] for th in thresholds}
    liver_dices = []
    argmax_tumor_dices = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            targets = batch['segmentation'].numpy()
            outputs = model(images)
            preds_argmax = torch.argmax(outputs['seg_logits'], dim=1).cpu().numpy()
            probs = F.softmax(outputs['seg_logits'], dim=1).cpu().numpy()
            
            prob_tumor = probs[:, 2:].sum(axis=1)

            for i in range(images.shape[0]):
                t = targets[i]
                l_d, _ = compute_binary_dice_jaccard(preds_argmax[i] >= 1, t >= 1)
                liver_dices.append(l_d)

                if (t >= 2).sum() > 0:
                    d_argmax, _ = compute_binary_dice_jaccard(preds_argmax[i] >= 2, t >= 2)
                    argmax_tumor_dices.append(d_argmax)
                    
                    for th in thresholds:
                        p_tum = (prob_tumor[i] >= th) & (preds_argmax[i] >= 1)
                        t_tum = (t >= 2)
                        d, _ = compute_binary_dice_jaccard(p_tum, t_tum)
                        res_per_thresh[th].append(d)

    print(f"Mean Liver Dice (Argmax): {np.mean(liver_dices):.2f}%")
    print(f"Mean Tumor Dice (Argmax): {np.mean(argmax_tumor_dices):.2f}%")
    print("\nTumor Dice across calibrated thresholds (within Liver):")
    for th in thresholds:
        print(f"  Threshold {th:0.2f} -> Mean Tumor Dice: {np.mean(res_per_thresh[th]):.2f}%")

if __name__ == '__main__':
    eval_thresholds()
