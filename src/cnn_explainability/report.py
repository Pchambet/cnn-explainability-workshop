"""Self-contained HTML report (``site/index.html``) built from ``results/``.

Charts are Plotly (loaded from jsDelivr) so readers can hover the exact values; static
figures are the PNGs in ``docs/figures/``, copied next to the page.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from string import Template

from cnn_explainability import figures, masks
from cnn_explainability.config import AMBER, FIGURES, RESULTS, SITE, SLATE, TEAL

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
COLOURS = {"vgg16_flat": TEAL, "vgg16_gap": AMBER, "eigenfaces": SLATE, "pixels": SLATE}
DASH = {"vgg16_flat": "solid", "vgg16_gap": "solid", "eigenfaces": "dash", "pixels": "dot"}


def _pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%"


def _rank1_traces(summary: list[dict]) -> list[dict]:
    conditions = list(masks.CONDITIONS)
    traces = []
    for rep, (label, *_rest) in figures.REPRESENTATIONS.items():
        rows = [figures._pick(summary, rep, c) for c in conditions]
        mean = [r["rank1_mean"] for r in rows]
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": label,
                "x": [figures.SHORT[c] for c in conditions],
                "y": mean,
                "error_y": {
                    "type": "data",
                    "symmetric": False,
                    "array": [r["rank1_hi"] - m for r, m in zip(rows, mean, strict=True)],
                    "arrayminus": [m - r["rank1_lo"] for r, m in zip(rows, mean, strict=True)],
                    "thickness": 1,
                    "width": 3,
                },
                "line": {"color": COLOURS[rep], "width": 2, "dash": DASH[rep]},
                "marker": {"size": 8},
                "hovertemplate": "%{x}<br>rank-1 %{y:.1%}<extra>" + label + "</extra>",
            }
        )
    return traces


def _tar_traces(summary: list[dict]) -> list[dict]:
    conditions = list(masks.CONDITIONS)
    traces = []
    for rep in ("vgg16_flat", "vgg16_gap", "eigenfaces"):
        rows = [figures._pick(summary, rep, c) for c in conditions]
        traces.append(
            {
                "type": "bar",
                "name": figures.REPRESENTATIONS[rep][0],
                "x": [figures.SHORT[c] for c in conditions],
                "y": [r["tar_at_far_mean"] for r in rows],
                "marker": {"color": COLOURS[rep]},
                "hovertemplate": "%{x}<br>TAR %{y:.1%}<extra></extra>",
            }
        )
    return traces


def _table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build() -> None:
    figures.build_all()
    summary = figures.load_summary()
    lfw = json.loads((RESULTS / "lfw.json").read_text())
    portrait = json.loads((RESULTS / "portrait.json").read_text())
    latency = json.loads((RESULTS / "latency.json").read_text())
    facts = key_facts(summary, lfw, portrait, latency)

    page = Template(TEMPLATE.read_text()).substitute(
        plotly=PLOTLY,
        rank1=json.dumps(_rank1_traces(summary)),
        tar=json.dumps(_tar_traces(summary)),
        chance=lfw["chance_rank1"],
        rank1_table=rank1_table(summary),
        attacker_table=attacker_table(summary),
        v1_table=v1_table(portrait),
        lat_table=latency_table(latency),
        **facts,
    )
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "index.html").write_text(page)
    shutil.copytree(FIGURES, SITE / "figures", dirs_exist_ok=True)
    print(f"wrote {SITE / 'index.html'}")


def peak_text(peaks: dict) -> str:
    cam = figures.ZONE_NAMES[peaks["gradcam_block5_conv3"]]
    occ = figures.ZONE_NAMES[peaks["occlusion"]]
    if cam == occ:
        return f"Grad-CAM at block5 and occlusion both peak on the {cam}"
    return f"Grad-CAM at block5 peaks on the {cam}, occlusion on the {occ}"


def key_facts(summary: list[dict], lfw: dict, portrait: dict, latency: dict) -> dict:
    """Every number quoted in prose, computed in one place."""

    def r(rep: str, cond: str, metric: str = "rank1_mean", gallery: str = "clean") -> float:
        return figures._pick(summary, rep, cond, gallery)[metric]

    eyes = lfw["matcher_occlusion"]["regions"]["eyes"]
    scan = lfw["matcher_occlusion"]["eye_bar_scan"]
    flat = latency["models"]["vgg16_classifier"]
    return {
        "n_ids": lfw["n_identities"],
        "n_images": f"{lfw['n_images']:,}",
        "n_probes": f"{lfw['n_probes_per_draw']:,}",
        "n_draws": lfw["n_draws"],
        "n_fit_ids": lfw["baseline_fit_identities"],
        "n_fit_images": f"{lfw['baseline_fit_images']:,}",
        "chance_pct": _pct(lfw["chance_rank1"], 1),
        "flat_clean": _pct(r("vgg16_flat", "none"), 1),
        "flat_eye": _pct(r("vgg16_flat", "eye_bar"), 1),
        "flat_eye_kept": _pct(r("vgg16_flat", "eye_bar") / r("vgg16_flat", "none")),
        "flat_eye_x_chance": f"{r('vgg16_flat', 'eye_bar') / lfw['chance_rank1']:.1f}",
        "flat_eye_matched": _pct(r("vgg16_flat", "eye_bar", gallery="matched"), 1),
        "gap_eye_matched": _pct(r("vgg16_gap", "eye_bar", gallery="matched"), 1),
        "flat_eyes_nose": _pct(r("vgg16_flat", "eyes_nose_bar"), 1),
        "flat_mild": _pct(r("vgg16_flat", "blur_mild"), 1),
        "flat_face": _pct(r("vgg16_flat", "face_box"), 1),
        "flat_face_matched": _pct(r("vgg16_flat", "face_box", gallery="matched"), 1),
        "flat_auc_clean": f"{r('vgg16_flat', 'none', 'auc_mean'):.2f}",
        "flat_auc_eye": f"{r('vgg16_flat', 'eye_bar', 'auc_mean'):.2f}",
        "tar_flat_eye": _pct(r("vgg16_flat", "eye_bar", "tar_at_far_mean"), 1),
        "flat_strong": _pct(r("vgg16_flat", "blur_strong"), 1),
        "gap_clean": _pct(r("vgg16_gap", "none"), 1),
        "gap_eye": _pct(r("vgg16_gap", "eye_bar"), 1),
        "eig_clean": _pct(r("eigenfaces", "none"), 1),
        "eig_eye": _pct(r("eigenfaces", "eye_bar"), 1),
        "px_clean": _pct(r("pixels", "none"), 1),
        "tar_flat_clean": _pct(r("vgg16_flat", "none", "tar_at_far_mean")),
        "tar_gap_clean": _pct(r("vgg16_gap", "none", "tar_at_far_mean")),
        "v1_tar": _pct(r("vgg16_flat", "none", "tar_fixed_mean")),
        "v1_far": _pct(r("vgg16_flat", "none", "far_fixed_mean"), 1),
        "eye_sens": _pct(eyes["sensitivity_share"]),
        "eye_area": _pct(eyes["area_share"]),
        "bar_max": _pct(scan["max"]["sensitivity_share"]),
        "bar_min": _pct(scan["min"]["sensitivity_share"]),
        "target": portrait["target_class"],
        "target_p": f"{portrait['top5'][0]['p']:.2f}",
        "peak_text": peak_text(portrait["peak_zones"]),
        "occ_raised": _pct(portrait["occlusion_raise"]["share_of_pixels_raised"]),
        "occ_median": f"{portrait['occlusion_raise']['median_change']:+.2f}",
        "occ_max_raise": f"{portrait['occlusion_raise']['max_raise']:.2f}",
        "eye_bar_p": f"{_masked(portrait, 'eye_bar')['p_original_class']:.2f}",
        "corr_b1": f"{portrait['filters']['block1_conv2']['mean_abs_correlation']:.2g}",
        "corr_b3": f"{portrait['filters']['block3_conv3']['mean_abs_correlation']:.2g}",
        "corr_b5": f"{portrait['filters']['block5_conv3']['mean_abs_correlation']:.2g}",
        "tar_eig_clean": _pct(r("eigenfaces", "none", "tar_at_far_mean"), 1),
        "lat_b1": f"{flat['batch1_min_ms']:.0f}",
        "lat_b32": f"{flat['batch32_min_per_image_ms']:.0f}",
        "lat_load": f"{latency['load_average_1min_at_start']:.0f}",
        "lat_load_end": f"{latency['load_average_1min_at_end']:.0f}",
        "lat_cpus": latency["logical_cpus"],
        "lat_threads": latency["threads"],
        "params": f"{portrait['params_total'] / 1e6:.1f}",
        "dense_share": _pct(portrait["params_dense"] / portrait["params_total"]),
    }


def _cell(r: dict, metric: str = "rank1") -> str:
    return f"{_pct(r[metric + '_mean'], 1)} [{_pct(r[metric + '_lo'])}, {_pct(r[metric + '_hi'])}]"


def rank1_table(summary: list[dict]) -> str:
    rows = [
        [label] + [_cell(figures._pick(summary, rep, c)) for c in masks.CONDITIONS]
        for rep, (label, *_rest) in figures.REPRESENTATIONS.items()
    ]
    return _table(["Representation", *figures.SHORT.values()], rows)


def attacker_table(summary: list[dict]) -> str:
    rows = []
    for rep in ("vgg16_flat", "vgg16_gap", "eigenfaces"):
        for gallery, name in (("clean", "clean enrolment"), ("matched", "masked enrolment")):
            label = f"{figures.REPRESENTATIONS[rep][0]}, {name}"
            rows.append(
                [label]
                + [
                    f"{figures._pick(summary, rep, c, gallery)['rank1_mean']:.1%}"
                    for c in masks.CONDITIONS
                ]
            )
    return _table(["Attacker", *figures.SHORT.values()], rows)


def _masked(portrait: dict, condition: str) -> dict:
    return next(m for m in portrait["masked_predictions"] if m["condition"] == condition)


def v1_table(portrait: dict) -> str:
    preds = portrait["masked_predictions"]
    target = portrait["target_class"]
    # The p(original class) column only adds information when some mask changes the label.
    changed = any(m["top1"] != target for m in preds)
    rows = [
        [figures.SHORT[m["condition"]], m["top1"], f"{m['top1_p']:.3f}"]
        + ([f"{m['p_original_class']:.3f}"] if changed else [])
        for m in preds
    ]
    header = ["Portrait", "ImageNet top-1", "p(top-1)"] + ([f"p({target})"] if changed else [])
    return _table(header, rows)


def latency_table(latency: dict) -> str:
    rows = [
        [
            name.replace("_", " "),
            f"{m['params'] / 1e6:.1f} M",
            f"{m['float32_mb']:.0f} MB",
            f"{m['batch1_min_ms']:.0f} ms",
            f"{m['batch1_median_ms']:.0f} ms",
            f"{m['batch32_min_per_image_ms']:.0f} ms",
        ]
        for name, m in latency["models"].items()
    ]
    header = [
        "Model",
        "Parameters",
        "float32 weights",
        "Batch 1, best",
        "Batch 1, median",
        "Per image in batch 32, best",
    ]
    return _table(header, rows)


TEMPLATE = Path(__file__).with_name("report_template.html")
