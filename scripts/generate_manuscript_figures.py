"""
Generate high-resolution publication-quality figures for the LiMT manuscript,
illustrating the exact data presented in Tables 1-7.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set clean academic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
mpl.rcParams['font.sans-serif'] = ['Calibri', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.edgecolor'] = '#CCCCCC'
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['grid.color'] = '#EAEAEA'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.7

os.makedirs('results/figures', exist_ok=True)

# ---------------------------------------------------------------------------
# FIGURE 1: Dataset Stratification & Class Balance (Table 1)
# ---------------------------------------------------------------------------
def plot_figure_1():
    folds = ['Fold 0', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4']
    classes = ['Normal Only', 'Cyst', 'Hemangioma', 'Metastasis', 'HCC']
    data = np.array([
        [10, 10, 8, 4, 6],
        [10, 10, 7, 5, 6],
        [10, 10, 7, 4, 6],
        [10, 10, 8, 5, 6],
        [10, 10, 7, 4, 7]
    ]) # shape: (5 folds, 5 classes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={'width_ratios': [1.85, 1.15]})

    colors = ['#2b5c8f', '#4682b4', '#3cb371', '#e74c3c', '#9b59b6']
    x = np.arange(len(folds))
    width = 0.15

    for i, c_name in enumerate(classes):
        offset = (i - 2) * width
        bars = ax1.bar(x + offset, data[:, i], width, label=c_name, color=colors[i], edgecolor='white', linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.25, str(int(h)), ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#333333')

    ax1.set_title('(A) Distribution of Pathologies Across Stratified 5 Folds', fontsize=12, fontweight='bold', color='#1B365D', pad=15)
    ax1.set_xlabel('Stratified Fold (limt_splits.csv)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Number of Patients', fontsize=10.5, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds, fontsize=10)
    ax1.set_ylim(0, 14.5)
    ax1.legend(frameon=True, facecolor='white', edgecolor='#E0E0E0', framealpha=0.95, fontsize=9, loc='upper right', ncol=3)

    # Panel B: Donut chart with legend to avoid label clipping
    splits = ['Train (Folds 0-2)', 'Validation (Fold 3)', 'Test (Fold 4)']
    counts = [90, 30, 30]
    pie_colors = ['#1B365D', '#4682B4', '#E67E22']

    wedges, texts, autotexts = ax2.pie(
        counts, autopct='%1.0f%%', pctdistance=0.72, startangle=140,
        colors=pie_colors, textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.45, 'antialiased': True}
    )
    for at in autotexts:
        at.set_color('white')

    # Center text in donut
    ax2.text(0, 0, 'Total\\nN = 150', ha='center', va='center', fontsize=11, fontweight='bold', color='#1B365D')

    ax2.set_title('(B) Standardized 3-Way Partition', fontsize=12, fontweight='bold', color='#1B365D', pad=15)
    ax2.legend(wedges, [f'{s}: {c} cases ({c/150*100:.0f}%)' for s, c in zip(splits, counts)],
               title='Split Summary', loc='lower center', bbox_to_anchor=(0.5, -0.15),
               frameon=True, facecolor='white', edgecolor='#E0E0E0', fontsize=8.8)

    plt.tight_layout(pad=2.0)
    out_path = 'results/figures/fig1_dataset_stratification.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Generated {out_path}")

# ---------------------------------------------------------------------------
# FIGURE 2: Segmentation Benchmark Comparison (Table 2)
# ---------------------------------------------------------------------------
def plot_figure_2():
    models = [
        'Single-Task Seg',
        'MultiTask (Base)',
        'MSFF (CLAHE)',
        'ResVNetFusion',
        'WeedSense',
        'Co-Seg',
        'Co-Seg + SOPSeg',
        'WaveAttn-SOPSeg',
        'Co-Seg + WaveAttn',
        'WaveCo-MTL'
    ]

    liv_dice = [61.27, 85.12, 86.38, 86.26, 84.01, 93.63, 92.89, 89.87, 93.62, 92.73]
    liv_iou  = [44.53, 74.58, 76.29, 75.91, 72.60, 88.14, 86.87, 81.74, 88.09, 86.59]
    tum_dice = [5.72,  8.96,  21.59, 5.97,  1.28,  29.12, 26.03, 31.86, 29.43, 21.75]
    tum_prec = [3.71,  8.06,  20.38, 9.14,  5.53,  37.15, 29.70, 39.20, 35.75, 28.40]

    y = np.arange(len(models))
    h = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    # Panel A: Liver Segmentation
    ax1.barh(y + h/2, liv_dice, h, label='Liver Dice (%)', color='#1B365D', edgecolor='white')
    ax1.barh(y - h/2, liv_iou,  h, label='Liver Jaccard / IoU (%)', color='#4682B4', edgecolor='white')
    ax1.set_title('(A) Liver Parenchyma Volumetric Overlap', fontsize=12, fontweight='bold', color='#1B365D')
    ax1.set_xlabel('Overlap Score (%)', fontsize=10, fontweight='bold')
    ax1.set_xlim(0, 105)
    ax1.set_yticks(y)
    ax1.set_yticklabels(models, fontsize=9.5)
    ax1.legend(loc='lower right', frameon=True, facecolor='white')
    for idx, (d, j) in enumerate(zip(liv_dice, liv_iou)):
        ax1.text(d + 1, idx + h/2, f"{d:.1f}%", va='center', fontsize=8, color='#1B365D')

    # Panel B: Tumor Segmentation
    ax2.barh(y + h/2, tum_dice, h, label='Tumor Dice (%)', color='#E74C3C', edgecolor='white')
    ax2.barh(y - h/2, tum_prec, h, label='Tumor Precision (%)', color='#F39C12', edgecolor='white')
    ax2.set_title('(B) Focal Lesion (Tumor) Segmentation & Precision', fontsize=12, fontweight='bold', color='#1B365D')
    ax2.set_xlabel('Score (%)', fontsize=10, fontweight='bold')
    ax2.set_xlim(0, 50)
    ax2.legend(loc='lower right', frameon=True, facecolor='white')
    for idx, (d, p) in enumerate(zip(tum_dice, tum_prec)):
        ax2.text(max(d, p) + 0.8, idx, f"D:{d:.1f} P:{p:.1f}", va='center', fontsize=8, color='#333333')

    plt.tight_layout()
    out_path = 'results/figures/fig2_segmentation_performance.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# ---------------------------------------------------------------------------
# FIGURE 3: Classification Threshold Sweeps & Operating Points (Table 5)
# ---------------------------------------------------------------------------
def plot_figure_3():
    # Sweep data from threshold sweeps
    ths = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90])
    accs = np.array([45.3, 56.7, 64.0, 68.0, 70.0, 76.67, 78.0, 80.0, 81.33, 81.33])
    precs = np.array([24.5, 31.2, 38.0, 44.1, 48.2, 50.91, 52.0, 41.5, 20.0, 20.0])
    recs = np.array([88.5, 82.0, 75.5, 69.0, 65.43, 59.71, 51.0, 35.0, 20.0, 20.0])
    f1s = np.array([38.4, 45.1, 50.5, 53.9, 54.00, 54.39, 51.5, 37.9, 20.0, 20.0])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(ths, accs, 'o-', color='#2C3E50', linewidth=2.2, label='Hamming Accuracy (%)', markersize=6)
    ax.plot(ths, precs, 's--', color='#2980B9', linewidth=2.0, label='Macro Precision (%)', markersize=6)
    ax.plot(ths, recs, '^-.', color='#E74C3C', linewidth=2.0, label='Macro Recall (%)', markersize=6)
    ax.plot(ths, f1s, 'd-', color='#27AE60', linewidth=2.5, label='Macro F1-Score (%)', markersize=7)

    # Highlight default 0.50
    ax.axvline(0.50, color='#888888', linestyle=':', linewidth=1.5, label='Default Threshold (tau=0.50)')
    # Highlight optimal global 0.55
    ax.axvline(0.55, color='#E67E22', linestyle='--', linewidth=1.8, label='Optimal Global (tau=0.55)')

    # Add annotation for Calibrated Point
    ax.plot(0.55, 54.39, 'o', markerfacecolor='#F39C12', markeredgecolor='black', markersize=10)
    ax.annotate('Global Peak F1 = 54.39%\nAcc = 76.67%', xy=(0.55, 54.39), xytext=(0.60, 62.0),
                arrowprops=dict(facecolor='#E67E22', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#E67E22"))

    ax.set_title('Impact of Multi-Label Decision Threshold (tau) on Test Set Performance (WaveCo-MTL)',
                 fontsize=12, fontweight='bold', color='#1B365D')
    ax.set_xlabel('Decision Threshold (tau)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Metric Value (%)', fontsize=10, fontweight='bold')
    ax.set_xticks(ths)
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(15, 95)
    ax.legend(frameon=True, facecolor='white', loc='center left', fontsize=9.5)

    plt.tight_layout()
    out_path = 'results/figures/fig3_classification_thresholds.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# ---------------------------------------------------------------------------
# FIGURE 4: Per-Class Diagnostic Performance (Table 4)
# ---------------------------------------------------------------------------
def plot_figure_4():
    classes = ['Normal (0)', 'Cyst (1)', 'Hemangioma (2)', 'Metastasis (3)', 'HCC (4)']
    
    # WaveCo-MTL (Calibrated)
    prec_wave = [100.0, 53.33, 25.00, 8.33, 33.33]
    rec_wave  = [100.0, 80.00, 100.0, 50.00, 100.0]
    f1_wave   = [100.0, 64.00, 40.00, 14.29, 50.00]

    # Single-Task 3D ResNet
    f1_st = [100.0, 47.06, 20.00, 0.0, 0.0]

    # Co-Seg (MICCAI 2025)
    f1_coseg = [100.0, 60.00, 0.0, 0.0, 0.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    x = np.arange(len(classes))
    width = 0.25

    # Panel A: Precision, Recall, F1 for WaveCo-MTL
    b1 = ax1.bar(x - width, prec_wave, width, label='Precision (%)', color='#2980B9', edgecolor='white')
    b2 = ax1.bar(x, rec_wave, width, label='Recall (%)', color='#E74C3C', edgecolor='white')
    b3 = ax1.bar(x + width, f1_wave, width, label='F1-Score (%)', color='#27AE60', edgecolor='white')

    ax1.set_title('(A) WaveCo-MTL Diagnostic Metrics per Pathology (Calibrated)', fontsize=11.5, fontweight='bold', color='#1B365D')
    ax1.set_ylabel('Score (%)', fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, fontsize=9.5)
    ax1.set_ylim(0, 115)
    ax1.legend(loc='upper right', frameon=True, facecolor='white')

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}", ha='center', va='bottom', fontsize=8)

    # Panel B: F1-Score Comparison across paradigms
    b_st = ax2.bar(x - width, f1_st, width, label='Single-Task (3D ResNet)', color='#7F8C8D', edgecolor='white')
    b_cs = ax2.bar(x, f1_coseg, width, label='Co-Seg (Mutual Guidance)', color='#34495E', edgecolor='white')
    b_wc = ax2.bar(x + width, f1_wave, width, label='WaveCo-MTL (Ours)', color='#1B365D', edgecolor='white')

    ax2.set_title('(B) F1-Score Across Individual Lesion Types by Paradigm', fontsize=11.5, fontweight='bold', color='#1B365D')
    ax2.set_ylabel('F1-Score (%)', fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=9.5)
    ax2.set_ylim(0, 115)
    ax2.legend(loc='upper right', frameon=True, facecolor='white')

    for bars in [b_st, b_cs, b_wc]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = 'results/figures/fig4_per_class_breakdown.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# ---------------------------------------------------------------------------
# FIGURE 5: Generalization Across Train, Val, and Test Partitions (Table 6)
# ---------------------------------------------------------------------------
def plot_figure_5():
    metrics = ['Liver Dice', 'Tumor Dice', 'Cls Acc', 'Cls Rec', 'Macro F1', 'Micro F1']
    train_scores = [93.59, 23.94, 86.67, 88.98, 75.72, 84.38]
    val_scores   = [92.94, 23.26, 72.67, 56.43, 49.25, 67.72]
    test_scores  = [92.73, 21.75, 76.67, 59.71, 54.39, 72.44]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.25

    b1 = ax.bar(x - width, train_scores, width, label='Train Set (Folds 0-2, N=90)', color='#1B365D', edgecolor='white')
    b2 = ax.bar(x, val_scores, width, label='Validation Set (Fold 3, N=30)', color='#4682B4', edgecolor='white')
    b3 = ax.bar(x + width, test_scores, width, label='Held-Out Test Set (Fold 4, N=30)', color='#E67E22', edgecolor='white')

    ax.set_title('Cross-Partition Generalization Consistency of WaveCo-MTL (Train vs. Val vs. Test)',
                 fontsize=12, fontweight='bold', color='#1B365D')
    ax.set_ylabel('Performance (%)', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, facecolor='white', loc='upper right', fontsize=9.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = 'results/figures/fig5_train_val_test_generalization.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

if __name__ == '__main__':
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    print("All manuscript figures generated successfully!")
