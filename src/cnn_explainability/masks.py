"""Image obfuscations used to "anonymise" a face, applied to uint8 RGB arrays.

Regions are fractional boxes ``(top, bottom, left, right)`` so one layout works at any
resolution. LFW images are funneled (aligned), so a single fixed layout places the bar
over the eyes of every face; the layout was read off the mean face of the evaluation subset
(the hero figure draws the boxes on that mean face).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FaceLayout:
    eyes: Box
    eyes_nose: Box
    face: Box


LFW_LAYOUT = FaceLayout(
    eyes=(0.32, 0.47, 0.15, 0.85),
    eyes_nose=(0.32, 0.64, 0.15, 0.85),
    face=(0.18, 0.82, 0.17, 0.83),
)
# Layout on the single portrait (assets/portrait.jpg), full-width bars as in the original workshop.
PORTRAIT_LAYOUT = FaceLayout(
    eyes=(0.30, 0.39, 0.0, 1.0),
    eyes_nose=(0.30, 0.45, 0.0, 1.0),
    face=(0.22, 0.60, 0.26, 0.74),
)

# Condition name -> human label, in the order they are reported.
CONDITIONS = {
    "none": "No mask",
    "eye_bar": "Eye bar",
    "eyes_nose_bar": "Eyes + nose bar",
    "blur_mild": "Mild face blur",
    "blur_strong": "Strong face blur",
    "face_box": "Face blacked out",
}
# Gaussian blur sigma as a fraction of the image width (4% and 13%; 13% is the original
# workshop's sigma of 30 px on a 224 px image).
BLUR_SIGMA = {"blur_mild": 0.04, "blur_strong": 0.134}


def box_slices(shape: tuple[int, ...], box: Box) -> tuple[slice, slice]:
    """Row and column slices of a fractional box on an array of the given shape."""
    h, w = shape[:2]
    top, bottom, left, right = box
    return slice(round(h * top), round(h * bottom)), slice(round(w * left), round(w * right))


def black_box(image: np.ndarray, box: Box) -> np.ndarray:
    out = image.copy()
    out[box_slices(image.shape, box)] = 0
    return out


def blur_box(image: np.ndarray, box: Box, sigma_frac: float) -> np.ndarray:
    """Gaussian-blur the region (blurring the whole image first avoids a sharp-edged halo)."""
    sigma = sigma_frac * image.shape[1]
    blurred = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(sigma)))
    out = image.copy()
    rows, cols = box_slices(image.shape, box)
    out[rows, cols] = blurred[rows, cols]
    return out


def apply(image: np.ndarray, condition: str, layout: FaceLayout = LFW_LAYOUT) -> np.ndarray:
    if condition == "none":
        return image.copy()
    if condition == "eye_bar":
        return black_box(image, layout.eyes)
    if condition == "eyes_nose_bar":
        return black_box(image, layout.eyes_nose)
    if condition == "face_box":
        return black_box(image, layout.face)
    if condition in BLUR_SIGMA:
        return blur_box(image, layout.face, BLUR_SIGMA[condition])
    raise ValueError(f"unknown condition {condition!r}; expected one of {list(CONDITIONS)}")


def apply_batch(images: np.ndarray, condition: str, layout: FaceLayout = LFW_LAYOUT) -> np.ndarray:
    return np.stack([apply(img, condition, layout) for img in images])
