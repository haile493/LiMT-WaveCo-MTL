import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def generate_splits(csv_path="LiMT Dataset/cls_labels.csv", output_path="LiMT Dataset/limt_splits.csv", seed=1337):
    df = pd.read_csv(csv_path)
    
    # Create multi-label signature string
    df['strata'] = df[['1', '2', '3', '4']].astype(str).agg('-'.join, axis=1)
    
    # Group rare strata (< 5 occurrences) to maintain balanced multi-label distribution
    strata_counts = df['strata'].value_counts()
    df['strata_grouped'] = df['strata'].apply(lambda x: x if strata_counts[x] >= 5 else 'other_multilesion')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['strata_grouped'])):
        df.loc[val_idx, 'fold'] = fold
    
    # Also add standard train/val/test split column for 3-way partition:
    # Folds 0, 1, 2 -> Train (90 cases, 60%)
    # Fold 3 -> Validation (30 cases, 20%)
    # Fold 4 -> Test (30 cases, 20%)
    def assign_split(fold):
        if fold in [0, 1, 2]:
            return 'train'
        elif fold == 3:
            return 'val'
        else:
            return 'test'
            
    df['split_3way'] = df['fold'].apply(assign_split)
    
    df.to_csv(output_path, index=False)
    print(f"Saved splits to {output_path}")
    print("\nSummary per fold:")
    for f in range(5):
        sub = df[df['fold'] == f]
        print(f"Fold {f}: total={len(sub)}, normal={(sub['strata']=='0-0-0-0').sum()}, "
              f"cyst={sub['1'].sum()}, hemangioma={sub['2'].sum()}, "
              f"metastasis={sub['3'].sum()}, hcc={sub['4'].sum()}")
              
    print("\nSummary for 3-way split:")
    for s in ['train', 'val', 'test']:
        sub = df[df['split_3way'] == s]
        print(f"Split {s}: total={len(sub)}, normal={(sub['strata']=='0-0-0-0').sum()}, "
              f"cyst={sub['1'].sum()}, hemangioma={sub['2'].sum()}, "
              f"metastasis={sub['3'].sum()}, hcc={sub['4'].sum()}")

if __name__ == "__main__":
    generate_splits()
