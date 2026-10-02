"""One-shot face identification and verification, evaluated from similarity matrices.

The protocol mirrors how an eye bar would be attacked in practice: the attacker holds one
*unmasked* reference photo per person (the gallery) and receives a *masked* photo (the probe).
Every metric is computed from ``S[i, j] = cos(probe embedding of image i, clean embedding of
image j)`` over the whole subset, so a new enrolment draw costs an index lookup, not a
forward pass.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def similarity(probes: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between two embedding sets (rows = images)."""
    return l2_normalize(probes) @ l2_normalize(references).T


def one_shot_split(labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One random enrolment image per identity; every other image is a probe."""
    gallery = np.array([rng.choice(np.flatnonzero(labels == c)) for c in np.unique(labels)])
    probes = np.setdiff1d(np.arange(len(labels)), gallery)
    return gallery, probes


def rank1_accuracy(sim: np.ndarray, labels: np.ndarray, gallery: np.ndarray, probes: np.ndarray):
    """Share of probes whose most similar enrolled image belongs to the right person."""
    scores = sim[np.ix_(probes, gallery)]
    predicted = labels[gallery][scores.argmax(axis=1)]
    return float((predicted == labels[probes]).mean())


def genuine_impostor(sim: np.ndarray, labels: np.ndarray, gallery: np.ndarray, probes: np.ndarray):
    """Probe-vs-enrolment scores split into same-person and different-person pairs."""
    scores = sim[np.ix_(probes, gallery)]
    same = labels[probes][:, None] == labels[gallery][None, :]
    return scores[same], scores[~same]


def roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """P(genuine score > impostor score), ties counted half (Mann-Whitney U / n1 n2)."""
    scores = np.concatenate([genuine, impostor])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    # average ranks for ties
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    n1, n0 = len(genuine), len(impostor)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def threshold_at_far(impostor: np.ndarray, far: float) -> float:
    """Smallest threshold whose false-accept rate (impostor >= t) does not exceed ``far``."""
    k = int(np.floor(far * len(impostor)))
    ordered = np.sort(impostor)[::-1]
    return float(np.nextafter(ordered[k], np.inf)) if k < len(ordered) else -np.inf


def rates_at(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> tuple[float, float]:
    """(true-accept rate, false-accept rate) of the rule ``score >= threshold``."""
    return float((genuine >= threshold).mean()), float((impostor >= threshold).mean())


def evaluate(
    sims: dict[str, np.ndarray],
    labels: np.ndarray,
    n_draws: int,
    seed: int,
    far: float = 0.01,
    fixed_threshold: float | None = None,
) -> list[dict]:
    """Metrics per condition and per enrolment draw.

    All conditions share the same enrolment draws, so differences between conditions are
    paired and not blurred by which photo happened to be enrolled. The verification
    threshold is calibrated on the *unmasked* condition at the target FAR and then frozen,
    as a deployed system would be.
    """
    rng = np.random.default_rng(seed)
    clean = next(iter(sims))
    rows = []
    for draw in range(n_draws):
        gallery, probes = one_shot_split(labels, rng)
        gen, imp = genuine_impostor(sims[clean], labels, gallery, probes)
        calibrated = threshold_at_far(imp, far)
        for condition, sim in sims.items():
            gen, imp = genuine_impostor(sim, labels, gallery, probes)
            tar, far_obs = rates_at(gen, imp, calibrated)
            row = {
                "condition": condition,
                "draw": draw,
                "rank1": rank1_accuracy(sim, labels, gallery, probes),
                "auc": roc_auc(gen, imp),
                "tar_at_far": tar,
                "far_observed": far_obs,
            }
            if fixed_threshold is not None:
                row["tar_fixed"], row["far_fixed"] = rates_at(gen, imp, fixed_threshold)
            rows.append(row)
    return rows
