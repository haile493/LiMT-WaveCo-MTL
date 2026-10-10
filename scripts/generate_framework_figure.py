"""
Generate publication-quality architectural framework diagram for WaveCo-MTL
(Figure 2 of the manuscript).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

os.makedirs('results/figures', exist_ok=True)

def create_framework_diagram():
    fig = plt.figure(figsize=(17, 9.6), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette - Professional Academic AI
    BG_COLOR = "#FFFFFF"
    fig.patch.set_facecolor(BG_COLOR)

    NAVY = "#1B365D"
    SLATE = "#34495E"
    BLUE = "#2980B9"
    LIGHT_BLUE = "#EBF5FB"
    BORDER_BLUE = "#7FB3D5"
    
    GREEN = "#229954"
    LIGHT_GREEN = "#EAFAF1"
    BORDER_GREEN = "#82E0AA"

    RED = "#C0392B"
    LIGHT_RED = "#FDEDEC"
    BORDER_RED = "#F1948A"

    ORANGE = "#D35400"
    LIGHT_ORANGE = "#FEF9E7"
    BORDER_ORANGE = "#FAD7A0"

    PURPLE = "#7D3C98"
    LIGHT_PURPLE = "#F4ECF7"
    BORDER_PURPLE = "#BB8FCE"

    GRAY_FILL = "#F8F9FA"
    GRAY_BORDER = "#D5D8DC"

    # Helper function: draw rounded box
    def draw_box(x, y, w, h, title, subtitle="", fill=LIGHT_BLUE, border=BORDER_BLUE, 
                 title_color=NAVY, title_size=10.5, sub_size=8.5, lw=1.5, radius=1.0):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius},rounding_size={radius}",
                             facecolor=fill, edgecolor=border, linewidth=lw, antialiased=True)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', 
                    fontsize=title_size, fontweight='bold', color=title_color)
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', 
                    fontsize=sub_size, color=SLATE)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', 
                    fontsize=title_size, fontweight='bold', color=title_color)

    # Helper function: draw section header bar
    def draw_section_hdr(x, y, w, h, text, fill=NAVY, text_color="white"):
        hdr = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.6",
                             facecolor=fill, edgecolor="none", antialiased=True)
        ax.add_patch(hdr)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=10.5, fontweight='bold', color=text_color)

    # Helper function: draw arrow
    def draw_arrow(x1, y1, x2, y2, color=SLATE, lw=1.6, style='->', rad=0.0, dashed=False):
        ls = '--' if dashed else '-'
        conn = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                connectionstyle=conn,
                                arrowstyle=style,
                                color=color,
                                linewidth=lw,
                                linestyle=ls,
                                mutation_scale=14,
                                zorder=5)
        ax.add_patch(arrow)

    # =========================================================================
    # 0. MAIN TITLE & BANNER
    # =========================================================================
    title_box = FancyBboxPatch((1.5, 93.0), 97.0, 5.5, boxstyle="round,pad=0.5,rounding_size=0.8",
                               facecolor="#1B365D", edgecolor="none")
    ax.add_patch(title_box)
    ax.text(50, 96.2, "WaveCo-MTL: Wavelet-Attentive Collaborative Multi-Task Learning Framework",
            ha='center', va='center', fontsize=15, fontweight='bold', color='white')
    ax.text(50, 94.2, "Joint 3D Liver & Lesion Segmentation with Segmentation-Guided Multi-Label Pathological Classification",
            ha='center', va='center', fontsize=10.5, color="#D6EAF8")

    # =========================================================================
    # 1. SECTION PANELS BACKGROUNDS
    # =========================================================================
    # Panel 1: Input & Shared Residual Encoder (x: 1.5 -> 28)
    bg_enc = FancyBboxPatch((1.5, 2.5), 26.5, 89.0, boxstyle="round,pad=0.8,rounding_size=1.2",
                            facecolor="#F9FBFD", edgecolor="#D4E6F1", linewidth=1.2)
    ax.add_patch(bg_enc)
    draw_section_hdr(2.5, 87.5, 24.5, 3.0, "(A) Shared 3D Residual Encoder", fill="#2874A6")

    # Panel 2: Wavelet & SOPSeg Bottleneck (x: 29.5 -> 53.5)
    bg_bot = FancyBboxPatch((29.5, 2.5), 24.0, 89.0, boxstyle="round,pad=0.8,rounding_size=1.2",
                            facecolor="#FDFEFE", edgecolor="#E5E8E8", linewidth=1.2)
    ax.add_patch(bg_bot)
    draw_section_hdr(30.5, 87.5, 22.0, 3.0, "(B) Edge Wavelet & SOPSeg", fill="#6C3483")

    # Panel 3: Co-Seg Dual-Stream Collaborative Decoder (x: 55.0 -> 76.5)
    bg_dec = FancyBboxPatch((55.0, 42.0), 43.5, 49.5, boxstyle="round,pad=0.8,rounding_size=1.2",
                            facecolor="#FDFEFE", edgecolor="#D5D8DC", linewidth=1.2)
    ax.add_patch(bg_dec)
    draw_section_hdr(56.0, 87.5, 41.5, 3.0, "(C) Dual-Stream Collaborative Segmentation (Co-Seg)", fill="#1E8449")

    # Panel 4: MFP & Classification Head (x: 55.0 -> 98.5, bottom)
    bg_cls = FancyBboxPatch((55.0, 2.5), 43.5, 38.0, boxstyle="round,pad=0.8,rounding_size=1.2",
                            facecolor="#FCF3CF", edgecolor="#F9E79F", linewidth=1.2)
    ax.add_patch(bg_cls)
    draw_section_hdr(56.0, 36.5, 41.5, 3.0, "(D) Masked Feature Pooling (MFP) & Multi-Label Classification", fill="#B7950B")

    # =========================================================================
    # 2. ENCODER BLOCKS (PANEL A)
    # =========================================================================
    # Input Volume
    draw_box(4.5, 76.0, 20.5, 8.5, "3D Abdominal CT Volume", 
             "Input: B × 1 × 64 × 64 × 64 (HU: [-100, 250])", 
             fill="#E8F8F5", border="#A3E4D7", title_color="#117A65", title_size=10.5)

    draw_arrow(14.75, 76.0, 14.75, 72.5, lw=2.0)

    # Stage 1
    draw_box(4.5, 63.5, 20.5, 8.5, "Encoder Stage 1 (Stem + ResBlock)",
             "C = 24 | Resolution: 64³\nPre-Activation InstNorm + LeakyReLU",
             fill=LIGHT_BLUE, border=BORDER_BLUE)
    draw_arrow(14.75, 63.5, 14.75, 59.5, lw=2.0)

    # Stage 2
    draw_box(4.5, 50.5, 20.5, 8.5, "Encoder Stage 2 (Strided ResBlock)",
             "C = 48 | Resolution: 32³\nSpatial Downsample (Stride 2)",
             fill=LIGHT_BLUE, border=BORDER_BLUE)
    draw_arrow(14.75, 50.5, 14.75, 46.5, lw=2.0)

    # Stage 3
    draw_box(4.5, 37.5, 20.5, 8.5, "Encoder Stage 3 (ResBlock)",
             "C = 96 | Resolution: 16³\nHigh-level semantic feature extraction",
             fill=LIGHT_BLUE, border=BORDER_BLUE)
    draw_arrow(14.75, 37.5, 14.75, 33.5, lw=2.0)

    # Stage 4 / Bottleneck
    draw_box(4.5, 24.5, 20.5, 8.5, "Encoder Stage 4 (Bottleneck)",
             "C = 192 | Resolution: 8³\nDeep latent spatial representation",
             fill=LIGHT_BLUE, border=BORDER_BLUE)

    # ResBlock Architecture Legend inset at bottom of Panel A
    leg_box = FancyBboxPatch((3.5, 4.5), 22.5, 17.5, boxstyle="round,pad=0.5,rounding_size=0.8",
                            facecolor="white", edgecolor="#BDC3C7", linewidth=1.0)
    ax.add_patch(leg_box)
    ax.text(14.75, 20.2, "Residual Block Detail", ha='center', va='center', 
            fontsize=9.5, fontweight='bold', color=NAVY)
    ax.text(14.75, 15.0, "x ──> [InstanceNorm 3D]\n    ──> [LeakyReLU (α=0.01)]\n    ──> [Conv3D (3×3×3)]\n    ──> [InstanceNorm + LeakyReLU]\n    ──> [Conv3D (3×3×3)] + x",
            ha='center', va='center', fontsize=8.0, family='monospace', color=SLATE)
    ax.text(14.75, 7.5, "(+) Identity Skip Connection", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color="#2980B9")

    # =========================================================================
    # 3. WAVELET & SOPSEG BOTTLENECK (PANEL B)
    # =========================================================================
    # Edge Branch (HaarDWT3D) branching from Stage 3
    draw_arrow(25.0, 41.75, 31.0, 68.0, color=ORANGE, lw=2.0, rad=-0.2)
    ax.text(27.5, 57.0, "Stage 3 Features\n(16³)", fontsize=8.0, fontweight='bold', color=ORANGE, ha='center')

    draw_box(31.5, 63.0, 20.0, 11.0, "3D Haar Wavelet (HaarDWT3D)",
             "8 Subbands Decomposition (8³):\n• 1 Low-Freq (LLL)\n• 7 High-Freq: LLH, LHL, LHH,\n  HLL, HLH, HHL, HHH (Edges)",
             fill=LIGHT_ORANGE, border=BORDER_ORANGE, title_color=ORANGE, title_size=9.8, sub_size=7.8)

    # 1x1 conv edge projection
    draw_arrow(41.5, 63.0, 41.5, 57.5, color=ORANGE, lw=1.8)
    draw_box(32.5, 51.5, 18.0, 5.8, "1×1×1 Conv Edge Projection",
             "Preserves micro-structural contours", fill=LIGHT_ORANGE, border=BORDER_ORANGE,
             title_color=ORANGE, title_size=9.0, sub_size=7.5)

    # Stage 4 to Bottleneck Fusion
    draw_arrow(25.0, 28.75, 31.0, 37.0, color=NAVY, lw=2.0, rad=0.1)

    # Fusion node
    draw_box(31.5, 34.0, 20.0, 8.0, "Wavelet-Enhanced Bottleneck",
             "F_fused = Conv3D( [F_enc4; F_edge] )\nResolution: 8³ | C = 192",
             fill=LIGHT_PURPLE, border=BORDER_PURPLE, title_color=PURPLE, title_size=9.8, sub_size=8.0)

    # Arrow from Edge Projection into Fusion
    draw_arrow(41.5, 51.5, 41.5, 42.0, color=ORANGE, lw=2.0)
    ax.text(42.5, 46.5, "+ F_edge", fontsize=8.5, fontweight='bold', color=ORANGE)

    # SOPSeg Block below Fusion
    draw_arrow(41.5, 34.0, 41.5, 28.5, color=PURPLE, lw=2.0)
    draw_box(31.5, 12.0, 20.0, 16.0, "SOPSeg Texture Covariance",
             "Second-Order Bilinear Pooling:\n"
             "X ∈ ℝ^(C × N), N = 512 (8³)\n\n"
             "Covariance Matrix:\n"
             "C_cov = (1/N) X Xᵀ  (192×192)\n\n"
             "MLP Covariance Descriptor ──>\n"
             "Channel Attention Gating γ_tex",
             fill=LIGHT_PURPLE, border=BORDER_PURPLE, title_color=PURPLE, title_size=9.8, sub_size=7.6)

    # Bottleneck Output Feature Latent Tensor
    draw_arrow(41.5, 12.0, 41.5, 8.5, color=PURPLE, lw=1.8)
    draw_box(32.0, 4.0, 19.0, 4.5, "Calibrated Latent Tensor F_latent",
             "Shape: B × 192 × 8 × 8 × 8",
             fill="#E8DAEF", border=BORDER_PURPLE, title_color=PURPLE, title_size=9.0, sub_size=7.5)

    # =========================================================================
    # 4. DUAL-STREAM COLLABORATIVE SEGMENTATION DECODER (PANEL C)
    # =========================================================================
    # Latent tensor feeds Decoder
    draw_arrow(51.5, 38.0, 56.5, 58.0, color=NAVY, lw=2.2, rad=0.15)
    ax.text(53.5, 47.0, "Deep Features\n(8³)", fontsize=8.0, fontweight='bold', color=NAVY, ha='center')

    # U-Net 3D Decoder Block
    draw_box(56.5, 53.0, 11.5, 18.0, "3D Decoder\nBackbone",
             "Trilinear\nUpsampling +\nSkip Conns\nfrom Stages\n3, 2, 1\n(64³)",
             fill="#EAECEE", border="#ABB2B9", title_color=SLATE, title_size=9.5, sub_size=7.6)

    # Skip connections from encoder stages to decoder
    draw_arrow(25.0, 67.75, 62.25, 71.0, color="#7FB3D5", lw=1.4, rad=-0.35, dashed=True)
    ax.text(43.0, 79.0, "Skip Connections (64³, 32³, 16³)", fontsize=8.0, color="#2980B9", ha='center', style='italic')

    # Decoder splits into Stream 1 (Liver) and Stream 2 (Tumor)
    draw_arrow(68.0, 67.0, 72.0, 73.0, color=GREEN, lw=2.2)
    draw_arrow(68.0, 57.0, 72.0, 51.0, color=RED, lw=2.2)

    # Stream 1: Liver Stream
    draw_box(70.5, 70.0, 13.0, 9.0, "Liver Stream\nConv Head",
             "Residual Features\nResolution: 64³",
             fill=LIGHT_GREEN, border=BORDER_GREEN, title_color=GREEN, title_size=9.5, sub_size=7.5)

    # Stream 2: Tumor Stream
    draw_box(70.5, 46.5, 13.0, 9.0, "Tumor Stream\nConv Head",
             "Residual Features\nResolution: 64³",
             fill=LIGHT_RED, border=BORDER_RED, title_color=RED, title_size=9.5, sub_size=7.5)

    # Mutual Guidance Prompting Between Streams
    # Arrow Liver -> Tumor (Spatial containment)
    draw_arrow(78.5, 70.0, 78.5, 55.5, color=GREEN, lw=2.0, rad=0.15)
    ax.text(79.5, 63.0, "Liver Spatial Gate\n(Constrains tumor to liver)", 
            fontsize=7.5, fontweight='bold', color=GREEN, ha='left', va='center')

    # Arrow Tumor -> Liver (Lesion continuity cue)
    draw_arrow(75.5, 55.5, 75.5, 70.0, color=RED, lw=2.0, rad=0.15)
    ax.text(74.5, 63.0, "Lesion Cue\n(Organ continuity)", 
            fontsize=7.5, fontweight='bold', color=RED, ha='right', va='center')

    # Final Segmentation Predictions
    # Liver Output
    draw_arrow(83.5, 74.5, 87.0, 74.5, color=GREEN, lw=2.0)
    draw_box(87.0, 70.0, 10.0, 9.0, "P̂_liv Mask",
             "Liver Parenchyma\n(Sigmoid, 64³)",
             fill="#D5F5E3", border=GREEN, title_color=GREEN, title_size=9.5, sub_size=7.5)

    # Tumor Output
    draw_arrow(83.5, 51.0, 87.0, 51.0, color=RED, lw=2.0)
    draw_box(87.0, 46.5, 10.0, 9.0, "P̂_tum Mask",
             "Focal Lesions\n(Sigmoid, 64³)",
             fill="#FADBD8", border=RED, title_color=RED, title_size=9.5, sub_size=7.5)

    # =========================================================================
    # 5. MASKED FEATURE POOLING (MFP) & CLASSIFICATION HEAD (PANEL D)
    # =========================================================================
    # Column 1 (x: 56.5 to 69.0): Latent Feature Extraction & Pooling
    # Arrow from F_latent (Panel B) into Column 1 poolers
    draw_arrow(51.0, 6.25, 56.5, 27.5, color=PURPLE, lw=1.6, rad=-0.08)
    draw_arrow(51.0, 6.25, 56.5, 18.5, color=PURPLE, lw=1.6, rad=-0.04)
    draw_arrow(51.0, 6.25, 56.5, 9.0, color=PURPLE, lw=1.8)
    ax.text(53.2, 13.5, "F_latent\n(8³)", fontsize=8.0, fontweight='bold', color=PURPLE, ha='center')

    # Top: Global Avg Pool
    draw_box(56.5, 23.5, 12.5, 7.5, "f_avg = GAP(F)",
             "Global Average Pooling\nWhole-Volume (192-d)",
             fill="#FEF9E7", border="#F9E79F", title_color="#7D6608", title_size=8.5, sub_size=6.8)

    # Middle: Global Max Pool
    draw_box(56.5, 14.5, 12.5, 7.5, "f_max = GMP(F)",
             "Global Max Pooling\nPeak Activation (192-d)",
             fill="#FEF9E7", border="#F9E79F", title_color="#7D6608", title_size=8.5, sub_size=6.8)

    # Bottom: Masked Lesion Pool (MFP)
    draw_box(56.5, 4.5, 12.5, 8.5, "f_lesion = MFP",
             "∑ (W_tum ⊙ F_latent)\n──────────────────\n(∑ W_tum + ε)  (192-d)",
             fill="#FDEDEC", border=BORDER_RED, title_color=RED, title_size=8.5, sub_size=6.5)

    # Downsample P_tum from Panel C into Soft Mask W_tum in Column 2
    draw_arrow(87.0, 46.5, 77.0, 31.0, color=RED, lw=1.8, rad=0.1)
    ax.text(85.0, 41.5, "Trilinear\nDownsample (8³)", fontsize=7.2, color=RED, ha='left', va='center', fontweight='bold')

    # Column 2 (x: 71.0 to 83.5): Soft Mask & Composite Representation
    # Top: Soft Mask W_tum
    draw_box(71.0, 22.5, 12.5, 8.5, "Soft Mask W_tum",
             "Downsampled (8³)\nLesion Probability Gate",
             fill="#FADBD8", border=RED, title_color=RED, title_size=8.8, sub_size=6.8)

    # Spatial Gate arrow from W_tum down to f_lesion
    draw_arrow(73.5, 22.5, 68.5, 13.0, color=RED, lw=1.8, rad=-0.12)
    ax.text(73.5, 18.0, "Soft Prior", fontsize=7.0, fontweight='bold', color=RED, ha='left', va='center')

    # Bottom: Composite f_cls
    draw_box(71.0, 4.5, 12.5, 13.5, "Composite f_cls",
             "Multi-Task Descriptor:\n[ f_avg ;\n  f_max ;\n  f_lesion ]\n\nLength: 576-d",
             fill="#FCF3CF", border="#F39C12", title_color="#B7950B", title_size=8.8, sub_size=6.8)

    # Inflow arrows from poolers into Composite f_cls
    draw_arrow(69.0, 27.25, 71.0, 16.0, color="#7D6608", lw=1.5)
    draw_arrow(69.0, 18.25, 71.0, 13.0, color="#7D6608", lw=1.5)
    draw_arrow(69.0, 8.75, 71.0, 8.75, color=RED, lw=1.8)

    # Column 3 (x: 85.5 to 97.5): Classifier Head & Final Multi-Label Output
    # Top: MLP Classifier Head
    draw_box(85.5, 19.5, 12.0, 11.5, "MLP Classifier",
             "Linear(576, 128)\n"
             "LeakyReLU (α=0.01)\n"
             "Linear(128, 5)\n"
             "Sigmoid Output\n(Dropout p = 0.3)",
             fill="white", border="#B7950B", title_color=NAVY, title_size=8.5, sub_size=6.8)

    # Arrow from Composite f_cls to MLP Classifier
    draw_arrow(83.5, 11.25, 85.5, 24.0, color=NAVY, lw=2.0, rad=-0.1)

    # Arrow from MLP Classifier down to Probabilities
    draw_arrow(91.5, 19.5, 91.5, 15.5, color=NAVY, lw=2.0)

    # Bottom: 5 Multi-Label Diagnostic Probabilities
    draw_box(85.5, 4.5, 12.0, 11.0, "5 Class Output p̂",
             "• Normal Parenchyma\n• Cyst\n• Hemangioma\n• Metastasis\n• HCC\n(Thresholds τ*)",
             fill="#D4EFDF", border=GREEN, title_color="#145A32", title_size=8.5, sub_size=6.8)

    # =========================================================================
    # 6. LOSS OBJECTIVES SUMMARY BAR
    # =========================================================================
    loss_box = FancyBboxPatch((1.5, 0.4), 97.0, 1.8, boxstyle="round,pad=0.2,rounding_size=0.4",
                             facecolor="#EAEDED", edgecolor="#BDC3C7", linewidth=1.0)
    ax.add_patch(loss_box)
    ax.text(50, 1.3, "Joint Multi-Task Loss:   ℒ_total = ℒ_seg + α · ℒ_cls   |   ℒ_seg = λ_liv · (ℒ_BCE + ℒ_Dice)_liv + λ_tum · (ℒ_BCE + ℒ_Dice)_tum (λ_tum = 3.0)   |   ℒ_cls = Weighted Multi-Label BCE",
            ha='center', va='center', fontsize=8.8, fontweight='bold', color=NAVY)

    out_path = 'results/figures/fig2_waveco_mtl_framework.png'
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Successfully generated framework diagram: {out_path}")

if __name__ == '__main__':
    create_framework_diagram()
