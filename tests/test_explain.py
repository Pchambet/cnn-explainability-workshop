"""Explainability methods checked on tiny hand-built models with a known answer."""

import keras
import numpy as np
import pytest

from cnn_explainability.explain import (
    grad_cam,
    histogram_entropy,
    maximise_activation,
    mean_abs_correlation,
    occlusion_map,
    region_mean,
)


def test_occlusion_finds_the_only_region_the_score_depends_on():
    # Score = mean brightness of the top-left 16x16 corner: occluding anywhere else is free.
    image = np.full((64, 64, 3), 255, np.uint8)

    def score(batch):
        return batch[:, :16, :16].mean(axis=(1, 2, 3)) / 255.0

    heat, base = occlusion_map(image, score, patch=16, stride=8)
    assert base == pytest.approx(1.0)
    assert heat[:8, :8].min() > 0.4
    assert heat[32:, 32:].max() == 0.0
    assert region_mean(heat, (0, 0.25, 0, 0.25)) > region_mean(heat, (0.5, 1, 0.5, 1))


def test_occlusion_covers_the_last_row_and_column():
    image = np.zeros((40, 40, 3), np.uint8)
    seen = []

    def score(batch):
        seen.append(batch.copy())
        return np.zeros(len(batch))

    occlusion_map(image, score, patch=10, stride=10, fill=7)
    patched = np.concatenate(seen[1:])
    assert (patched[:, 30:, 30:] == 7).any()


def _blob_classifier():
    """Conv with a fixed brightness detector -> GAP -> one logit; class 0 = 'bright blob'."""
    inputs = keras.Input((32, 32, 1))
    conv = keras.layers.Conv2D(1, 3, padding="same", activation="relu", name="conv")
    pooled = keras.layers.GlobalAveragePooling2D()(conv(inputs))
    dense = keras.layers.Dense(1, name="head")
    model = keras.Model(inputs, dense(pooled))
    conv.set_weights([np.ones((3, 3, 1, 1), "float32") / 9, np.zeros(1, "float32")])
    dense.set_weights([np.ones((1, 1), "float32"), np.zeros(1, "float32")])
    return model


def test_grad_cam_peaks_on_the_evidence():
    x = np.zeros((1, 32, 32, 1), "float32")
    x[0, 20:26, 4:10] = 1.0
    cam = grad_cam(_blob_classifier(), x, "conv", class_index=0)
    assert cam.shape == (32, 32)
    peak = np.unravel_index(cam.argmax(), cam.shape)
    assert 19 <= peak[0] <= 26 and 3 <= peak[1] <= 10
    assert cam[:8, 20:].max() == 0.0


def test_activation_maximisation_recovers_a_known_colour_preference():
    # A single 1x1 filter that responds to red minus blue: the optimum is a red image.
    inputs = keras.Input((None, None, 3))
    conv = keras.layers.Conv2D(1, 1, name="redness")
    model = keras.Model(inputs, conv(inputs))
    conv.set_weights([np.array([1.0, 0.0, -1.0], "float32").reshape(1, 1, 3, 1), np.zeros(1)])
    img = maximise_activation(model, "redness", [0], size=16, steps=20, lr=1.0)[0]
    assert img[..., 0].mean() > img[..., 1].mean() > img[..., 2].mean()


def test_redundancy_and_entropy_metrics_hand_checked():
    a = np.linspace(0, 1, 48).reshape(4, 4, 3)
    assert mean_abs_correlation(np.stack([a, a, 1 - a])) == pytest.approx(1.0)
    assert histogram_entropy(np.full((8, 8, 3), 0.3)) == 0.0
    halves = np.concatenate([np.zeros((4, 8, 3)), np.ones((4, 8, 3))])
    assert histogram_entropy(halves) == pytest.approx(1.0)
