"""Static figures (PNG, 200 dpi) for the README, built only from files in ``results/``."""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from cnn_explainability import masks
from cnn_explainability.config import AMBER, FIGURES, GRID, INK, RESULTS, SLATE, TEAL

REPRESENTATIONS = {
    # key: (label, colour, linestyle, marker)
    "vgg16_flat": ("VGG16 pool5, flattened (v1 design)", TEAL, "-", "o"),
    "vgg16_gap": ("VGG16 pool5, averaged", AMBER, "-", "s"),
    "eigenfaces": ("Eigenfaces (PCA-100)", SLATE, "--", "^"),
    "pixels": ("Raw pixels", SLATE, ":", "v"),
}
SHORT = {
    "none": "No mask",
    "eye_bar": "Eye bar",
    "eyes_nose_bar": "Eyes + nose",
    "blur_mild": "Mild blur",
    "blur_strong": "Strong blur",
    "face_box": "Face blacked out",
}


ZONE_NAMES = {
    "forehead_hair": "forehead and hair",
    "eyes": "eyes",
    "nose": "nose",
    "mouth_chin": "mouth and chin",
    "neck_clothing": "neck and clothing",
    "background": "background",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": SLATE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": INK,
            "font.size": 10,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def load_summary() -> list[dict]:
    with (RESULTS / "lfw_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            k: (v if k in {"representation", "gallery", "condition"} else float(v))
            for k, v in r.items()
            if v != ""
        }
        for r in rows
    ]


def _pick(summary: list[dict], rep: str, cond: str, gallery: str = "clean") -> dict:
    return next(
        r
        for r in summary
        if r["representation"] == rep and r["condition"] == cond and r["gallery"] == gallery
    )


def mean_face() -> np.ndarray:
    """Average of the evaluation images: shows the mask geometry without showing anyone."""
    path = RESULTS / "lfw_mean_face.png"
    return np.asarray(Image.open(path).convert("RGB"))


def hero(summary: list[dict], meta: dict) -> None:
    conditions = list(masks.CONDITIONS)
    face = mean_face()
    fig = plt.figure(figsize=(10, 6.2))
    grid = fig.add_gridspec(2, len(conditions), height_ratios=[1, 2.6], hspace=0.08, wspace=0.05)
    for i, cond in enumerate(conditions):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(masks.apply(face, cond))
        ax.set_axis_off()
        ax.set_title(SHORT[cond], fontsize=9.5, loc="center", fontweight="normal")
    ax = fig.add_subplot(grid[1, :])
    x = np.arange(len(conditions))
    for rep, (label, colour, style, marker) in REPRESENTATIONS.items():
        rows = [_pick(summary, rep, c) for c in conditions]
        mean = np.array([r["rank1_mean"] for r in rows])
        lo = np.array([r["rank1_lo"] for r in rows])
        hi = np.array([r["rank1_hi"] for r in rows])
        ax.fill_between(x, lo, hi, color=colour, alpha=0.12, linewidth=0)
        ax.plot(x, mean, style, color=colour, marker=marker, lw=2, ms=6, label=label)
        if rep == "vgg16_flat":  # label the headline series only
            for xi, m in zip(x, mean, strict=True):
                ax.annotate(
                    f"{m:.1%}",
                    (xi, m),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=INK,
                    fontweight="bold",
                )
    chance = meta["chance_rank1"]
    ax.axhline(chance, color=SLATE, lw=1, ls=(0, (2, 3)))
    ax.text(
        len(conditions) - 1,
        chance,
        f"chance {chance:.1%}",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=SLATE,
    )
    ax.set_xticks(x, [SHORT[c] for c in conditions])
    ax.set_xlim(-0.6, len(conditions) - 0.4)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Rank-1 identification accuracy")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    flat = {c: _pick(summary, "vgg16_flat", c) for c in conditions}
    kept = flat["eye_bar"]["rank1_mean"] / flat["none"]["rank1_mean"]
    above_chance = all(r["rank1_lo"] > chance for r in flat.values())
    tail = "; no mask tested brings it down to chance" if above_chance else ""
    fig.suptitle(
        f"An eye bar keeps {kept:.0%} of a CNN's one-shot re-identification accuracy{tail}",
        x=0.125,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.125,
        0.935,
        f"{meta['n_identities']} LFW identities, one clean enrolment photo each, "
        f"{meta['n_probes_per_draw']:,} masked probes. Line = mean, band = 2.5-97.5% range over "
        f"{meta['n_draws']} enrolment draws. Masks shown on the average face.",
        fontsize=9,
        color=SLATE,
    )
    fig.savefig(FIGURES / "hero.png")
    plt.close(fig)


def verification(summary: list[dict], meta: dict) -> None:
    conditions = list(masks.CONDITIONS)
    reps = ["vgg16_flat", "vgg16_gap", "eigenfaces"]
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    width = 0.8 / len(reps)
    x = np.arange(len(conditions))
    for k, rep in enumerate(reps):
        label, colour, _, _ = REPRESENTATIONS[rep]
        rows = [_pick(summary, rep, c) for c in conditions]
        mean = np.array([r["tar_at_far_mean"] for r in rows])
        err = np.array(
            [
                [m - r["tar_at_far_lo"], r["tar_at_far_hi"] - m]
                for m, r in zip(mean, rows, strict=True)
            ]
        ).T
        pos = x + (k - (len(reps) - 1) / 2) * width
        ax.bar(pos, mean, width - 0.03, color=colour, label=label, zorder=2)
        ax.errorbar(
            pos, mean, yerr=err, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3
        )
    ax.set_xticks(x, [SHORT[c] for c in conditions])
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Genuine pairs accepted")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    clean = _pick(summary, "vgg16_flat", "none")["tar_at_far_mean"]
    eye = _pick(summary, "vgg16_flat", "eye_bar")["tar_at_far_mean"]
    ax.set_title(
        f"A threshold frozen at 1% false accepts stops accepting barred faces: "
        f"{clean:.1%} \u2192 {eye:.1%} for VGG16",
        fontsize=11,
    )
    fig.savefig(FIGURES / "verification.png")
    plt.close(fig)


def matcher_map(meta: dict) -> None:
    heat = np.load(RESULTS / "matcher_occlusion_mean.npy").astype("float32")
    face = mean_face()
    occ = meta["matcher_occlusion"]["regions"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.6), gridspec_kw={"wspace": 0.05})
    axes[0].imshow(face)
    axes[0].set_title("Average evaluation face", fontsize=10, fontweight="normal")
    axes[1].imshow(face)
    im = axes[1].imshow(np.clip(heat, 0, None), cmap="magma", alpha=0.65)
    h, w = heat.shape
    top, bottom, left, right = masks.LFW_LAYOUT.eyes
    axes[1].add_patch(
        Rectangle(
            (left * w, top * h),
            (right - left) * w,
            (bottom - top) * h,
            fill=False,
            ec="white",
            lw=1.5,
            ls="--",
        )
    )
    axes[1].text(right * w - 2, top * h - 3, "eye bar", color="white", fontsize=8.5, ha="right")
    axes[1].set_title(
        "Drop in genuine similarity when a patch is hidden", fontsize=10, fontweight="normal"
    )
    for ax in axes:
        ax.set_axis_off()
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cbar.set_label("cosine similarity drop", color=INK)
    eyes = occ["eyes"]
    fig.suptitle(
        f"The eye band holds {eyes['sensitivity_share']:.0%} of the matcher's sensitivity "
        f"for {eyes['area_share']:.0%} of the image",
        x=0.125,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(FIGURES / "matcher_occlusion.png")
    plt.close(fig)


def filters(maps: np.lib.npyio.NpzFile, stats: dict) -> None:
    layers = [k.removeprefix("filters_") for k in maps.files if k.startswith("filters_")]
    n = maps[f"filters_{layers[0]}"].shape[0]
    fig, axes = plt.subplots(len(layers), n, figsize=(n * 0.95, len(layers) * 1.15))
    for row, layer in enumerate(layers):
        for col in range(n):
            axes[row, col].imshow(maps[f"filters_{layer}"][col])
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
        corr = stats[layer]["mean_abs_correlation"]
        axes[row, 0].set_ylabel(
            f"{layer}\n|r| = {corr:.2f}",
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
            labelpad=6,
        )
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.suptitle(
        "Inputs that maximise the first 8 filters of three VGG16 layers",
        x=0.125,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    out = FIGURES / "filters.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    # Optimised images are noise-like and compress badly; a 256-colour palette keeps the
    # figure small with no visible loss.
    Image.open(out).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(
        out, optimize=True
    )


def portrait(maps: np.lib.npyio.NpzFile, result: dict) -> None:
    image = np.asarray(
        Image.open(RESULTS.parent / "assets" / "portrait.jpg")
        .convert("RGB")
        .resize((224, 224), Image.Resampling.BILINEAR)
    )
    cams = [k for k in maps.files if k.startswith("cam_")]
    panels = [("Input", None)] + [(f"Grad-CAM {k[4:]}", maps[k]) for k in cams]
    panels.append(("Occlusion (32 px patch)", maps["occlusion"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.9))
    for ax, (title, heat) in zip(axes, panels, strict=True):
        ax.imshow(image)
        if heat is not None:
            heat = np.clip(heat.astype("float32"), 0, None)
            ax.imshow(heat / (heat.max() + 1e-12), cmap="magma", alpha=0.55)
        ax.set_title(title, fontsize=9, fontweight="normal", loc="center")
        ax.set_axis_off()
    label = result["target_class"]
    p = result["top5"][0]["p"]
    cam_zone = ZONE_NAMES[result["peak_zones"]["gradcam_block5_conv3"]]
    occ_zone = ZONE_NAMES[result["peak_zones"]["occlusion"]]
    where = (
        f"Grad-CAM and occlusion both peak on the {cam_zone}"
        if cam_zone == occ_zone
        else f"Grad-CAM peaks on the {cam_zone}, occlusion on the {occ_zone}"
    )
    fig.suptitle(
        f"ImageNet VGG16 labels this portrait \u201c{label}\u201d (p = {p:.2f}); {where}",
        x=0.05,
        ha="left",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(FIGURES / "portrait_explanations.png")
    plt.close(fig)


def build_all() -> None:
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    meta = json.loads((RESULTS / "lfw.json").read_text())
    portrait_result = json.loads((RESULTS / "portrait.json").read_text())
    hero(summary, meta)
    verification(summary, meta)
    matcher_map(meta)
    with np.load(RESULTS / "portrait_maps.npz") as maps:
        filters(maps, portrait_result["filters"])
        portrait(maps, portrait_result)
