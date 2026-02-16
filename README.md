# 🧠 CNN Filter Visualization & Grad-CAM — Face Recognition

> **TP Mise en œuvre** · Master TRIED · Conférence Ouverture Professionnelle

## What this is

A hands-on exploration of **CNN interpretability** applied to face recognition with VGG16:

| Part | Topic | Key Question |
|:---:|-------|-------------|
| 1 | **Filter Visualization** | What do CNN filters actually detect? |
| 2 | **Grad-CAM & Occlusion** | Is eye-masking enough for anonymization? |
| 3 | **One-Shot Learning** | KNN vs Neural Networks with minimal data? |
| 4 | **Production** | How to deploy a face recognition model? |

## Quick Start

```bash
# Install dependencies (requires uv)
uv sync

# Launch Jupyter
uv run jupyter lab
```

Then open **`TP_Reconnaissance_Faciale.ipynb`** and run cells sequentially.

## Project Structure

```
├── TP_Reconnaissance_Faciale.ipynb   ← Main notebook (run this)
├── test_face.jpg                     ← Test portrait image
├── pyproject.toml                    ← Dependencies (uv)
├── output_figures/                   ← Generated visualizations
│   ├── 01_filters.png
│   ├── 02_gradcam.png
│   ├── 03_occlusion.png
│   └── 04_cnil_mask.png
└── saved_model/                      ← Exported model (Part 4)
```

## Key Findings

- **CNN filters form a visual hierarchy**: edges → textures → face structures
- **Eye masking alone is insufficient** for anonymization — CNNs use nose, mouth, jawline
- **CNIL recommendations may be ethnically biased** (Shrutin et al., 2019)
- **KNN + transfer learning** is optimal for one-shot face recognition

## Tech Stack

- Python 3.11 · TensorFlow 2.20 · OpenCV · scikit-learn
- Package manager: [uv](https://github.com/astral-sh/uv)

## References

- [Visualizing what convnets learn](https://keras.io/examples/vision/visualizing_what_convnets_learn/) — Keras
- [Grad-CAM](https://keras.io/examples/vision/grad_cam/) — Keras
- [VGGFace](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/) — Oxford VGG
- [Deep Learning for Face Recognition: Pride or Prejudiced?](https://arxiv.org/pdf/1904.01219.pdf) — Shrutin et al., 2019
