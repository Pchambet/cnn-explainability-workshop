"""Three ways to look inside a CNN: activation maximisation, Grad-CAM and occlusion.

Each answers a different question. Activation maximisation shows what a filter *can*
respond to; Grad-CAM shows which feature-map locations the gradient of one output points to;
occlusion measures what actually happens to an output when a region is hidden. The first two
are cheap but rest on gradients; occlusion is slow but model-agnostic, which is why it is the
one reused to explain the face matcher.
"""

from __future__ import annotations

from collections.abc import Callable

import keras
import numpy as np
import tensorflow as tf
from PIL import Image


def maximise_activation(
    model: keras.Model,
    layer: str,
    filters: list[int],
    size: int = 128,
    steps: int = 30,
    lr: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """Gradient ascent in pixel space, one image per filter, optimised as a single batch.

    VGG16 has no batch normalisation, so images in a batch do not interact and each gradient
    only reflects its own filter. Gradients are L2-normalised per image so the step size does
    not depend on the activation scale, which differs by orders of magnitude across depths.
    Returns float images in [0, 1], shape (len(filters), size, size, 3).
    """
    extractor = keras.Model(model.input, model.get_layer(layer).output)
    rng = np.random.default_rng(seed)
    start = rng.uniform(0.5, 0.75, size=(len(filters), size, size, 3)).astype("float32")
    image = tf.Variable(start)
    one_hot = tf.one_hot(filters, extractor.output.shape[-1])  # (n, channels)

    @tf.function
    def step() -> None:
        with tf.GradientTape() as tape:
            act = extractor(image, training=False)[:, 2:-2, 2:-2, :]  # drop border artefacts
            per_image = tf.reduce_mean(act, axis=(1, 2))  # (n, channels)
            loss = tf.reduce_sum(per_image * one_hot)
        grads = tape.gradient(loss, image)
        image.assign_add(lr * tf.math.l2_normalize(grads, axis=(1, 2, 3)))

    for _ in range(steps):
        step()
    out = image.numpy()
    mean = out.mean(axis=(1, 2, 3), keepdims=True)
    std = out.std(axis=(1, 2, 3), keepdims=True)
    return np.clip((out - mean) / (std + 1e-5) * 0.15 + 0.5, 0.0, 1.0)


def histogram_entropy(image: np.ndarray, bins: int = 50) -> float:
    """Shannon entropy (bits) of the grey-level histogram: a crude complexity measure."""
    hist, _ = np.histogram(image.mean(axis=-1), bins=bins, range=(0.0, 1.0))
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def mean_abs_correlation(images: np.ndarray) -> float:
    """Mean |Pearson r| between all pairs of images: high means redundant visualisations."""
    corr = np.corrcoef(images.reshape(len(images), -1))
    off_diagonal = ~np.eye(len(images), dtype=bool)
    return float(np.abs(corr[off_diagonal]).mean())


def grad_cam(model: keras.Model, inputs: np.ndarray, layer: str, class_index: int) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., 2017) at the resolution of ``layer``, scaled to [0, 1]."""
    grad_model = keras.Model(model.input, [model.get_layer(layer).output, model.output])
    x = tf.convert_to_tensor(inputs)
    with tf.GradientTape() as tape:
        maps, preds = grad_model(x, training=False)
        score = preds[:, class_index]
    grads = tape.gradient(score, maps)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # one weight per channel
    cam = tf.nn.relu(tf.reduce_sum(maps[0] * weights, axis=-1)).numpy()
    return cam / (cam.max() + 1e-12)


def occlusion_map(
    image: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
    patch: int = 32,
    stride: int = 16,
    fill: int = 128,
    batch_size: int = 64,
) -> tuple[np.ndarray, float]:
    """Slide a grey patch over ``image`` and record how much ``score_fn`` drops.

    ``score_fn`` maps a uint8 batch (N, H, W, 3) to one score per image, so the same routine
    explains a classifier (class probability) and a matcher (cosine similarity to a reference).
    Each pixel receives the mean drop of every patch that covered it. Patches are evaluated in
    batches: one forward pass per patch would be dominated by call overhead.
    """
    h, w = image.shape[:2]
    base = float(score_fn(image[None])[0])
    positions = [(y, x) for y in _starts(h, patch, stride) for x in _starts(w, patch, stride)]
    drops = np.empty(len(positions))
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        batch = np.repeat(image[None], len(chunk), axis=0)
        for i, (y, x) in enumerate(chunk):
            batch[i, y : y + patch, x : x + patch] = fill
        drops[start : start + len(chunk)] = base - score_fn(batch)
    total, count = np.zeros((h, w)), np.zeros((h, w))
    for (y, x), drop in zip(positions, drops, strict=True):
        total[y : y + patch, x : x + patch] += drop
        count[y : y + patch, x : x + patch] += 1
    return total / np.maximum(count, 1), base


def _starts(length: int, patch: int, stride: int) -> list[int]:
    """Patch offsets along one axis; a last patch flush with the edge leaves no strip unseen."""
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def upsample(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a 2-D map to ``size`` = (height, width)."""
    img = Image.fromarray(heatmap.astype("float32"), mode="F")
    return np.asarray(img.resize((size[1], size[0]), Image.Resampling.BILINEAR))


def region_mean(heatmap: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """Mean of ``heatmap`` inside a fractional box (top, bottom, left, right)."""
    h, w = heatmap.shape
    top, bottom, left, right = box
    return float(heatmap[int(h * top) : int(h * bottom), int(w * left) : int(w * right)].mean())
