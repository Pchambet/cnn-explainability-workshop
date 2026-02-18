# 🧠 CNN Explainability Workshop

> **What do CNNs actually learn? And can we trust their decisions?**

A deep dive into CNN interpretability using VGG16 — from raw filter patterns to Grad-CAM heatmaps, with a quantitative challenge of CNIL anonymization guidelines.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Results at a Glance

### Part 1 — What CNN Filters Detect

![Filter visualization across 3 hierarchical levels of VGG16](output_figures/01_filters.png)

VGG16 filters form a **visual hierarchy**: simple edge detectors (block1) → complex textures (block3) → abstract structures (block5). This mirrors the human visual cortex (Hubel & Wiesel, 1962).

### Part 2 — Where the Network Looks (Grad-CAM)

![Grad-CAM heatmaps across 3 convolutional layers](output_figures/02_gradcam.png)

The model predicts **"jersey"** (clothing), not a face identity — because it was trained on ImageNet (objects), not faces. The Grad-CAM confirms this: attention is focused on the **shirt and neckline**, not facial features.

### Part 2b — Occlusion Sensitivity Analysis

![Occlusion sensitivity map showing critical image regions](output_figures/03_occlusion.png)

Systematic occlusion with a sliding patch (25×25 px, stride 14) reveals which regions are critical for classification. The sensitivity map **converges with Grad-CAM** findings, reinforcing the analysis.

### Part 2c — Challenging CNIL Anonymization Guidelines

![Comparison of 4 anonymization methods and their effect on CNN classification](output_figures/04_cnil_mask.png)

**Key finding**: eye masking alone (CNIL recommendation) is **insufficient** — the network still classifies correctly using nose, mouth, jawline, and skin texture. Only full-face Gaussian blur degrades CNN confidence significantly.

---

## Methodology

This project goes beyond visualization — it applies **quantitative analysis** at every step:

| Analysis | Method | Why it matters |
|----------|--------|----------------|
| **Filter diversity** | Shannon entropy + inter-filter correlation | Measures if filters are actually *different* from each other |
| **Grad-CAM by region** | ROI analysis (5 facial zones × 3 layers) | Quantifies *where* the network looks, not just visually |
| **Occlusion sensitivity** | Sliding patch with confidence δ measurement | Model-agnostic validation of Grad-CAM findings |
| **Anonymization efficacy** | 4 methods tested (eyes / eyes+nose / full face / Gaussian blur) | Experimental evidence against CNIL guidelines |

---

## Project Structure

| Part | Topic | Key Question |
|:---:|-------|-------------|
| 1 | **Filter Visualization** | What do CNN filters actually detect at each depth level? |
| 2 | **Grad-CAM & Occlusion** | Is CNIL eye-masking sufficient against modern CNNs? |
| 3 | **One-Shot Learning** | Can we build face recognition with a single photo per person? |
| 4 | **Production** | How to deploy, optimize, and comply with GDPR? |

```
├── TP_Reconnaissance_Faciale.ipynb   ← Main notebook (run this)
├── test_face.jpg                     ← Test portrait image
├── output_figures/                   ← Generated visualizations
│   ├── 01_filters.png
│   ├── 02_gradcam.png
│   ├── 03_occlusion.png
│   └── 04_cnil_mask.png
├── pyproject.toml                    ← Dependencies (uv)
└── README.md
```

## Quick Start

```bash
# Requires uv (https://github.com/astral-sh/uv)
uv sync
uv run jupyter lab
```

Then open `TP_Reconnaissance_Faciale.ipynb` and **Run All** (~8 min on CPU).

## Key Findings

1. **CNNs are not black boxes** — filter visualization + Grad-CAM provide complementary interpretability
2. **Interpretability depends on training domain** — same VGG16 looks at clothing (ImageNet) vs facial features (VGGFace)
3. **Eye masking is NOT enough** for anonymization — full-face blur is the only robust method
4. **Ethnic bias is structural** — datasets like LFW/VGGFace are >70% Caucasian, leading to biased anonymization effectiveness (Shrutin et al., 2019)
5. **KNN + transfer learning** is optimal for one-shot face recognition

## Tech Stack

Python 3.11 · TensorFlow 2.20 · OpenCV · scikit-learn · Matplotlib
Package manager: [uv](https://github.com/astral-sh/uv)

## References

- Selvaraju et al. (2017) — [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- Yosinski et al. (2014) — [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
- Shrutin et al. (2019) — [Deep Learning for Face Recognition: Pride or Prejudiced?](https://arxiv.org/abs/1904.01219)
- Keras tutorials: [Filter visualization](https://keras.io/examples/vision/visualizing_what_convnets_learn/) · [Grad-CAM](https://keras.io/examples/vision/grad_cam/)

## License

MIT
