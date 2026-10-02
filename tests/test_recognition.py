import numpy as np
import pytest

from cnn_explainability.recognition import (
    evaluate,
    genuine_impostor,
    one_shot_split,
    rank1_accuracy,
    rates_at,
    roc_auc,
    similarity,
    threshold_at_far,
)


def clustered_embeddings(n_ids=10, per_id=6, dim=32, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_ids, dim))
    labels = np.repeat(np.arange(n_ids), per_id)
    return centres[labels] + noise * rng.normal(size=(len(labels), dim)), labels


def test_split_enrols_exactly_one_image_per_identity():
    labels = np.array([0, 0, 1, 1, 1, 2, 2])
    gallery, probes = one_shot_split(labels, np.random.default_rng(0))
    assert sorted(labels[gallery]) == [0, 1, 2]
    assert len(np.intersect1d(gallery, probes)) == 0
    assert len(gallery) + len(probes) == len(labels)


def test_rank1_hand_checked():
    # probe 2 is closer to person 1's enrolment, probe 3 to person 0's: one hit out of two.
    labels = np.array([0, 1, 0, 1])
    sim = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 1.0, 0.0],
            [0.8, 0.2, 0.0, 1.0],
        ]
    )
    assert rank1_accuracy(sim, labels, np.array([0, 1]), np.array([2, 3])) == 0.5


def test_recovers_identities_on_well_separated_clusters():
    emb, labels = clustered_embeddings(noise=0.05)
    sim = similarity(emb, emb)
    gallery, probes = one_shot_split(labels, np.random.default_rng(1))
    assert rank1_accuracy(sim, labels, gallery, probes) == 1.0
    gen, imp = genuine_impostor(sim, labels, gallery, probes)
    assert roc_auc(gen, imp) == 1.0


def test_random_embeddings_score_near_chance():
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(20), 30)
    emb = rng.normal(size=(len(labels), 64))
    rows = evaluate({"none": similarity(emb, emb)}, labels, n_draws=10, seed=0)
    assert np.mean([r["rank1"] for r in rows]) == pytest.approx(1 / 20, abs=0.02)
    assert np.mean([r["auc"] for r in rows]) == pytest.approx(0.5, abs=0.02)


def test_auc_matches_pairwise_definition_with_ties():
    gen = np.array([0.9, 0.5, 0.5])
    imp = np.array([0.5, 0.1])
    pairs = [(g > i) + 0.5 * (g == i) for g in gen for i in imp]
    assert roc_auc(gen, imp) == pytest.approx(np.mean(pairs))


def test_threshold_respects_far_budget():
    rng = np.random.default_rng(3)
    imp = rng.normal(size=1000)
    t = threshold_at_far(imp, 0.01)
    _, far = rates_at(np.array([0.0]), imp, t)
    assert far <= 0.01
    assert far >= 0.009  # and is not needlessly conservative


def test_evaluate_freezes_threshold_on_clean_condition():
    emb, labels = clustered_embeddings(noise=0.5, seed=4)
    rng = np.random.default_rng(5)
    noisy = emb + 3.0 * rng.normal(size=emb.shape)
    sims = {"none": similarity(emb, emb), "noisy": similarity(noisy, emb)}
    rows = evaluate(sims, labels, n_draws=5, seed=0, fixed_threshold=0.5)
    by = {c: [r for r in rows if r["condition"] == c] for c in sims}
    assert np.mean([r["rank1"] for r in by["noisy"]]) < np.mean([r["rank1"] for r in by["none"]])
    assert all(r["far_observed"] <= 0.01 for r in by["none"])
    assert {"tar_fixed", "far_fixed"} <= set(rows[0])
