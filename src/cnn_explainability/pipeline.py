"""The three experiments behind the README, each writing small result files to ``results/``.

1. ``portrait``: what VGG16 sees in one portrait (filters, Grad-CAM, occlusion, and the
   ImageNet label under each mask: the measurement of the original workshop).
2. ``lfw``: does an eye bar stop one-shot re-identification? 62 LFW identities, one clean
   enrolment photo each, masked probes, 20 enrolment draws, four representations.
3. ``latency``: measured CPU latency and size of the models, replacing quoted figures.

Expensive intermediate arrays (embeddings) are cached in ``data/interim/`` so a re-run only
recomputes what is missing.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from cnn_explainability import explain, lfw, masks, recognition, vgg
from cnn_explainability.config import (
    DATA_RAW,
    LFW_BASELINE_MIN_IMAGES,
    LFW_MAX_IMAGES,
    LFW_MIN_IMAGES,
    N_ENROLMENT_DRAWS,
    PORTRAIT,
    RESULTS,
    ROOT,
    SEED,
)

INTERIM = ROOT / "data" / "interim"
FILTER_LAYERS = ["block1_conv2", "block3_conv3", "block5_conv3"]
CAM_LAYERS = ["block3_conv3", "block4_conv3", "block5_conv3"]
# Anatomical zones on the 224 x 224 portrait, (top, bottom, left, right) fractions, read off
# the resized image (the hair, eyes, nose, mouth, chin and suit collar of this particular photo).
PORTRAIT_ZONES = {
    "forehead_hair": (0.0, 0.30, 0.0, 1.0),
    "eyes": (0.30, 0.38, 0.25, 0.75),
    "nose": (0.38, 0.44, 0.38, 0.62),
    "mouth_chin": (0.44, 0.58, 0.30, 0.70),
    "neck_clothing": (0.58, 1.0, 0.0, 1.0),
}
V1_THRESHOLD = 0.5  # rejection threshold hard-coded in the original FaceRecognizer
GALLERY_PROBE_PAIR = (0, 1)  # file indices per identity used for the matcher occlusion maps
MATCHER_PATCH = 40  # px on the 180 x 160 LFW crop; stride 20 tiles it exactly


def _write_json(name: str, payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(payload, indent=2) + "\n")


def _load_portrait() -> np.ndarray:
    return vgg.resize(lfw.load_image(PORTRAIT, crop=None))


# --------------------------------------------------------------------------- portrait
def run_portrait() -> dict:
    model = vgg.load_vgg16(include_top=True)
    notop = vgg.load_vgg16(include_top=False)
    conv = sum(layer.count_params() for layer in model.layers if "conv" in layer.name)
    total = model.count_params()

    filters_viz, filter_stats = {}, {}
    for layer in FILTER_LAYERS:
        filters_viz[layer] = maximise_and_quantise(notop, layer, list(range(8)), size=128)
        sample = explain.maximise_activation(
            notop, layer, list(range(16)), size=64, steps=20, preprocess=vgg.preprocess_unit
        )
        entropies = [explain.histogram_entropy(img) for img in sample]
        filter_stats[layer] = {
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
            "mean_abs_correlation": explain.mean_abs_correlation(sample),
            "n_filters": len(sample),
        }

    image = _load_portrait()
    x = vgg.preprocess(image[None])
    probs = model(x, training=False).numpy()
    top = np.argsort(probs[0])[::-1][:5]
    labels = vgg.decode_top1(np.eye(1000)[top])
    top5 = [
        {"label": lab, "p": float(probs[0, i])} for (lab, _), i in zip(labels, top, strict=True)
    ]
    target = int(top[0])

    cams = {layer: explain.grad_cam(model, x, layer, target) for layer in CAM_LAYERS}
    cams = {layer: explain.upsample(cam, (224, 224)) for layer, cam in cams.items()}
    cam_zones = {
        layer: {zone: explain.region_mean(cam, box) for zone, box in PORTRAIT_ZONES.items()}
        for layer, cam in cams.items()
    }

    def class_prob(batch: np.ndarray) -> np.ndarray:
        return model(vgg.preprocess(batch), training=False).numpy()[:, target]

    occlusion, base = explain.occlusion_map(image, class_prob, patch=32, stride=16)
    occ_zones = {zone: explain.region_mean(occlusion, box) for zone, box in PORTRAIT_ZONES.items()}
    # Occlusion is not monotone on a classifier: hiding a patch can also raise the probability.
    occ_raise = {
        "share_of_pixels_raised": float((occlusion < 0).mean()),
        "median_change": float(-np.median(occlusion)),
        "max_raise": float(-occlusion.min()),
        "max_drop": float(occlusion.max()),
    }
    peaks = {
        "gradcam_block5_conv3": peak_zone(cams["block5_conv3"]),
        "occlusion": peak_zone(occlusion),
    }

    masked_preds = []
    for condition in masks.CONDITIONS:
        masked = masks.apply(image, condition, masks.PORTRAIT_LAYOUT)
        p = model(vgg.preprocess(masked[None]), training=False).numpy()
        label, score = vgg.decode_top1(p)[0]
        masked_preds.append(
            {
                "condition": condition,
                "top1": label,
                "top1_p": score,
                "p_original_class": float(p[0, target]),
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULTS / "portrait_maps.npz",
        **{f"filters_{k}": v for k, v in filters_viz.items()},
        **{f"cam_{k}": v.astype("float16") for k, v in cams.items()},
        occlusion=occlusion.astype("float16"),
    )
    payload = {
        "params_total": int(total),
        "params_conv": int(conv),
        "params_dense": int(total - conv),
        "filters": filter_stats,
        "top5": top5,
        "target_class": top5[0]["label"],
        "gradcam_zone_means": cam_zones,
        "occlusion_base_p": base,
        "occlusion_zone_means": occ_zones,
        "occlusion_raise": occ_raise,
        "peak_zones": peaks,
        "masked_predictions": masked_preds,
    }
    _write_json("portrait.json", payload)
    return payload


def peak_zone(heatmap: np.ndarray) -> str:
    """Name of the anatomical zone that contains the hottest pixel of a 224 x 224 map."""
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    fy, fx = y / heatmap.shape[0], x / heatmap.shape[1]
    for zone, (top, bottom, left, right) in PORTRAIT_ZONES.items():
        if top <= fy < bottom and left <= fx < right:
            return zone
    return "background"


def maximise_and_quantise(model, layer: str, filters: list[int], size: int) -> np.ndarray:
    images = explain.maximise_activation(
        model, layer, filters, size=size, preprocess=vgg.preprocess_unit
    )
    return (images * 255).astype("uint8")


# --------------------------------------------------------------------------- LFW
def _grey_small(images: np.ndarray, size: tuple[int, int] = (40, 45)) -> np.ndarray:
    """Grey 45 x 40 thumbnails, flattened: the input of the pixel and eigenface baselines."""
    from PIL import Image

    out = [np.asarray(Image.fromarray(img).convert("L").resize(size)) for img in images]
    flat = np.stack(out).reshape(len(images), -1).astype("float64")
    return flat - flat.mean(axis=1, keepdims=True)


def _vgg_embeddings(extractor, faces: lfw.FaceSet, condition: str) -> dict[str, np.ndarray]:
    cache = INTERIM / f"vgg_{condition}.npz"
    if cache.exists():
        with np.load(cache) as data:
            return {k: data[k].astype("float32") for k in data.files}
    t0 = time.perf_counter()
    emb = vgg.embed(extractor, masks.apply_batch(faces.images, condition))
    # Stored as float16 to halve the cache (62 MB per condition); return the stored values so
    # a first run and a cached re-run produce identical numbers.
    emb["flat"] = emb["flat"].astype("float16")
    INTERIM.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **emb)
    print(f"  embedded {condition}: {time.perf_counter() - t0:.0f} s", flush=True)
    return {k: v.astype("float32") for k, v in emb.items()}


def run_lfw() -> dict:
    from sklearn.decomposition import PCA

    folder = lfw.download(DATA_RAW)
    faces = lfw.load_subset(folder, LFW_MIN_IMAGES, cap=LFW_MAX_IMAGES)
    baseline_fit = lfw.load_subset(folder, LFW_BASELINE_MIN_IMAGES, max_images=19)
    assert not set(faces.names) & set(baseline_fit.names), "baseline must not see test ids"
    save_mean_face(faces)

    model = vgg.load_vgg16(include_top=True)
    extractor = vgg.embedding_model(model)
    pca = PCA(n_components=100, whiten=True, random_state=SEED)
    pca.fit(_grey_small(baseline_fit.images))

    def representations(condition: str) -> dict[str, np.ndarray]:
        emb = _vgg_embeddings(extractor, faces, condition)
        px = _grey_small(masks.apply_batch(faces.images, condition))
        return {
            "vgg16_flat": emb["flat"],
            "vgg16_gap": emb["gap"],
            "eigenfaces": pca.transform(px),
            "pixels": px,
        }

    # Two attackers. "clean": the enrolment photos are unmasked, as found online. "matched":
    # the attacker applies the same mask to the enrolment photos before comparing, which
    # costs nothing and removes the mismatch the mask creates.
    clean = representations("none")
    sims: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for condition in masks.CONDITIONS:
        probe = representations(condition)
        for rep in probe:
            for gallery, reference in (("clean", clean[rep]), ("matched", probe[rep])):
                sims.setdefault((rep, gallery), {})[condition] = recognition.similarity(
                    probe[rep], reference
                )

    rows = []
    for (rep, gallery), by_condition in sims.items():
        threshold = V1_THRESHOLD if rep == "vgg16_flat" else None
        for row in recognition.evaluate(
            by_condition, faces.labels, N_ENROLMENT_DRAWS, SEED, fixed_threshold=threshold
        ):
            rows.append({"representation": rep, "gallery": gallery, **row})
    _write_csv(RESULTS / "lfw_draws.csv", rows)

    summary = summarise(rows)
    _write_csv(RESULTS / "lfw_summary.csv", summary)

    sensitivity = matcher_occlusion(extractor, faces)
    payload = {
        "n_identities": len(faces.names),
        "images_per_identity": LFW_MAX_IMAGES,
        "n_images": len(faces.labels),
        "n_probes_per_draw": len(faces.labels) - len(faces.names),
        "n_draws": N_ENROLMENT_DRAWS,
        "chance_rank1": 1 / len(faces.names),
        "baseline_fit_identities": len(baseline_fit.names),
        "baseline_fit_images": len(baseline_fit.labels),
        "v1_threshold": V1_THRESHOLD,
        "matcher_occlusion": sensitivity,
    }
    _write_json("lfw.json", payload)
    return payload


def save_mean_face(faces: lfw.FaceSet) -> None:
    """The average evaluation image: shows the mask geometry without showing anyone."""
    from PIL import Image

    RESULTS.mkdir(parents=True, exist_ok=True)
    mean = faces.images.mean(axis=0).round().astype("uint8")
    Image.fromarray(mean).save(RESULTS / "lfw_mean_face.png")


def summarise(rows: list[dict]) -> list[dict]:
    """Mean and 2.5-97.5 percentile band over enrolment draws, per (representation, gallery,
    condition)."""
    ids = ("representation", "gallery", "condition")
    metrics = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in {*ids, "draw"}]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in ids), []).append(r)
    out = []
    for key, sub in groups.items():
        entry: dict = dict(zip(ids, key, strict=True))
        for m in metrics:
            if m not in sub[0]:
                continue
            values = np.array([r[m] for r in sub], dtype=float)
            entry[f"{m}_mean"] = float(values.mean())
            entry[f"{m}_lo"] = float(np.percentile(values, 2.5))
            entry[f"{m}_hi"] = float(np.percentile(values, 97.5))
        out.append(entry)
    return out


def matcher_occlusion(extractor, faces: lfw.FaceSet) -> dict:
    """Where does the *matcher* look? Occlusion maps of the genuine similarity.

    For every identity, photo 0 is enrolled and photo 1 is the probe; we slide a grey patch
    over the probe and record the drop in cosine similarity to the enrolment (flattened pool5
    embedding). Faces are aligned, so maps can be averaged across identities.
    """
    first = np.array([np.flatnonzero(faces.labels == c)[0] for c in range(len(faces.names))])
    enrol_idx, probe_idx = first + GALLERY_PROBE_PAIR[0], first + GALLERY_PROBE_PAIR[1]
    cache = INTERIM / "matcher_occlusion"
    cache.mkdir(parents=True, exist_ok=True)
    refs = None
    maps = []
    for k, i in enumerate(probe_idx):
        path = cache / f"{faces.names[k]}.npy"
        if not path.exists():  # one file per identity, so an interrupted run resumes
            if refs is None:
                flat = vgg.embed(extractor, faces.images[enrol_idx])["flat"]
                refs = recognition.l2_normalize(flat)

            def score(batch: np.ndarray, ref: np.ndarray = refs[k]) -> np.ndarray:
                emb = recognition.l2_normalize(vgg.embed(extractor, batch, batch_size=32)["flat"])
                return emb @ ref

            heat, _ = explain.occlusion_map(
                faces.images[i], score, patch=MATCHER_PATCH, stride=MATCHER_PATCH // 2
            )
            np.save(path, heat)
            print(f"  matcher occlusion {k + 1}/{len(probe_idx)}", flush=True)
        maps.append(np.load(path))
    maps = np.stack(maps)

    mean_map = maps.mean(axis=0)
    layout = masks.LFW_LAYOUT
    shares = region_shares(
        mean_map, {"eyes": layout.eyes, "eyes_nose": layout.eyes_nose, "face": layout.face}
    )
    # Is the eye band special? Slide a box of the eye bar's size down the face box and see
    # what the same amount of black would hide elsewhere.
    scan = bar_scan(mean_map, layout.eyes, rows=(layout.face[0], layout.face[1]))
    peak_row, peak_col = np.unravel_index(int(np.argmax(mean_map)), mean_map.shape)
    np.save(RESULTS / "matcher_occlusion_mean.npy", mean_map.astype("float16"))
    return {
        "n_identities": len(maps),
        "patch": MATCHER_PATCH,
        "stride": MATCHER_PATCH // 2,
        "regions": shares,
        "eye_bar_scan": scan,
        "peak": {
            "row_fraction": peak_row / mean_map.shape[0],
            "col_fraction": peak_col / mean_map.shape[1],
        },
    }


def region_shares(sensitivity: np.ndarray, boxes: dict[str, masks.Box]) -> dict:
    """Share of the total positive sensitivity, and of the image area, inside each box.

    Negative values (hiding a patch *raised* the score) are clipped: they are not evidence
    that the region carries identity.
    """
    positive = np.clip(sensitivity, 0, None)
    out = {}
    for name, box in boxes.items():
        region = positive[masks.box_slices(positive.shape, box)]
        out[name] = {
            "sensitivity_share": float(region.sum() / positive.sum()),
            "area_share": float(region.size / positive.size),
        }
    return out


def bar_scan(
    sensitivity: np.ndarray, bar: masks.Box, rows: tuple[float, float]
) -> dict[str, dict[str, float]]:
    """Slide ``bar`` vertically (same height and columns) between the ``rows`` fractions, one
    pixel row at a time, and return the positions where it hides the least and the most
    positive sensitivity."""
    positive = np.clip(sensitivity, 0, None)
    h = positive.shape[0]
    top, bottom, left, right = bar
    height = round(h * bottom) - round(h * top)
    cols = masks.box_slices(positive.shape, (0.0, 1.0, left, right))[1]
    row_mass = positive[:, cols].sum(axis=1) / positive.sum()
    starts = range(round(h * rows[0]), round(h * rows[1]) - height + 1)
    scores = {start: float(row_mass[start : start + height].sum()) for start in starts}
    lo, hi = min(scores, key=scores.get), max(scores, key=scores.get)
    return {
        "min": {"top": lo / h, "sensitivity_share": scores[lo]},
        "max": {"top": hi / h, "sensitivity_share": scores[hi]},
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


# --------------------------------------------------------------------------- latency
def run_latency(repeats: int = 30, threads: int = 3) -> dict:
    """Wall-clock latency on this machine's CPU (batch 1 and batch 32).

    On a shared machine the median mostly measures the neighbours; the minimum over repeats
    is the closer estimate of the model's own cost, so both are recorded.
    """
    model = vgg.load_vgg16(include_top=True)
    extractor = vgg.embedding_model(model)
    rng = np.random.default_rng(SEED)
    out = {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "system": platform.system(),
        "threads": threads,
        "load_average_1min_at_start": os.getloadavg()[0],
        "load_average_1min_at_end": None,
        "logical_cpus": os.cpu_count(),
        "repeats": repeats,
        "models": {},
    }
    for name, net in {"vgg16_classifier": model, "vgg16_pool5_extractor": extractor}.items():
        params = int(net.count_params())
        entry = {"params": params, "float32_mb": params * 4 / 1e6}
        for batch in (1, 32):
            x = vgg.preprocess(rng.integers(0, 256, (batch, 224, 224, 3), dtype=np.uint8))
            for _ in range(3):
                net(x, training=False)
            times = []
            for _ in range(repeats if batch == 1 else max(3, repeats // 6)):
                t0 = time.perf_counter()
                net(x, training=False)
                times.append(time.perf_counter() - t0)
            entry[f"batch{batch}_median_ms"] = float(np.median(times) * 1000)
            entry[f"batch{batch}_min_ms"] = float(np.min(times) * 1000)
            entry[f"batch{batch}_min_per_image_ms"] = float(np.min(times) * 1000 / batch)
        out["models"][name] = entry
    out["load_average_1min_at_end"] = os.getloadavg()[0]
    _write_json("latency.json", out)
    return out
