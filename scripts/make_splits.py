import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

df = pd.read_csv(r"LiMT Dataset\cls_labels.csv")
print("Total cases:", len(df))

# Check distribution of label combinations (signatures)
# For stratification with multi-label, we can create a composite label signature
df['strata'] = df[['1', '2', '3', '4']].astype(str).agg('-'.join, axis=1)
print("\nUnique multi-label combinations (strata):")
print(df['strata'].value_counts())

# We can perform iterative stratification or stratified grouping
# Let's inspect how many normal (0-0-0-0) vs diseased cases:
# 0-0-0-0 is 50 cases
# Let's see the distribution of each lesion class across folds
from sklearn.model_selection import StratifiedKFold

# Using iterative stratification or stratified k-fold on strata
# For strata with counts < 5, we can group them into a 'multi-lesion' or composite bin
strata_counts = df['strata'].value_counts()
df['strata_grouped'] = df['strata'].apply(lambda x: x if strata_counts[x] >= 5 else 'other_multilesion')
print("\nGrouped strata counts:")
print(df['strata_grouped'].value_counts())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['strata_grouped'])):
    df.loc[val_idx, 'fold'] = fold

print("\nFold distribution:")
for fold in range(5):
    sub = df[df['fold'] == fold]
    print(f"Fold {fold}: total={len(sub)}, normal={(sub['strata']=='0-0-0-0').sum()}, "
          f"cls1={sub['1'].sum()}, cls2={sub['2'].sum()}, cls3={sub['3'].sum()}, cls4={sub['4'].sum()}")
