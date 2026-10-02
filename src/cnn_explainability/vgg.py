"""VGG16 (ImageNet weights) loading, preprocessing and embedding extraction.

Keras downloads the weights once (~530 MB with the classifier head) to ``~/.keras/models``.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from cnn_explainability.config import IMG_SIZE


def limit_threads(n: int = 3) -> None:
    """Cap TensorFlow's CPU threads so the pipeline stays a polite neighbour on a laptop."""
    try:
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:  # already initialised: the first setting wins
        pass


def load_vgg16(include_top: bool = True) -> keras.Model:
    if include_top:
        return keras.applications.VGG16(weights="imagenet", include_top=True)
    return keras.applications.VGG16(weights="imagenet", include_top=False)


def resize(image: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize a uint8 RGB image to ``size x size`` (bilinear)."""
    if image.shape[:2] == (size, size):
        return image
    return np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR))


def preprocess(images: np.ndarray) -> np.ndarray:
    """uint8 batch (N, 224, 224, 3) -> VGG16 input (caffe-style BGR mean subtraction)."""
    return keras.applications.vgg16.preprocess_input(images.astype("float32"))


def decode_top1(probs: np.ndarray) -> list[tuple[str, float]]:
    decoded = keras.applications.vgg16.decode_predictions(probs, top=1)
    return [(label, float(score)) for _, label, score in (row[0] for row in decoded)]


def embedding_model(vgg: keras.Model) -> keras.Model:
    """Return pool5 activations (7 x 7 x 512): the feature map the one-shot matcher uses."""
    return keras.Model(vgg.input, vgg.get_layer("block5_pool").output)


def embed(extractor: keras.Model, images: np.ndarray, batch_size: int = 32) -> dict:
    """Two embeddings per image from one forward pass.

    ``flat`` is the 25,088-d flattened pool5 map (the design of the original workshop);
    ``gap`` is its 512-d global average, which discards *where* a feature fires.
    """
    flat, gap = [], []
    for start in range(0, len(images), batch_size):
        batch = np.stack([resize(img) for img in images[start : start + batch_size]])
        maps = extractor(preprocess(batch), training=False).numpy()
        flat.append(maps.reshape(len(maps), -1))
        gap.append(maps.mean(axis=(1, 2)))
    return {"flat": np.concatenate(flat), "gap": np.concatenate(gap)}
