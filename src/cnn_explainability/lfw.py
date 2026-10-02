"""Labeled Faces in the Wild (deep-funneled): idempotent download and a balanced subset loader.

Source: Huang et al. (2007), http://vis-www.cs.umass.edu/lfw/ , mirrored on figshare by
scikit-learn (same URL and checksum as ``sklearn.datasets.fetch_lfw_people``). LFW is a public
research benchmark of news photographs of public figures; it is not redistributed here.

The loader reads JPEGs directly instead of going through ``fetch_lfw_people`` because the
latter materialises every image as float32 in memory (about 1 GB for the subset we need).
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from cnn_explainability.config import DATA_RAW

URL = "https://ndownloader.figshare.com/files/5976015"
SHA256 = "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"
ARCHIVE_NAME = "lfw-funneled.tgz"
FOLDER_NAME = "lfw_funneled"

# Crop of the 250x250 funneled image: the face plus hair, ears and part of the neck. A wider
# context than scikit-learn's face-only default, so that masking the whole face still leaves
# something (hair, outline, clothing) a model could exploit.
CROP_BOX = (45, 40, 205, 220)  # (left, top, right, bottom) in pixels


@dataclass(frozen=True)
class FaceSet:
    """Images (uint8, N x H x W x 3), integer labels and the identity names."""

    images: np.ndarray
    labels: np.ndarray
    names: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(data_dir: Path = DATA_RAW) -> Path:
    """Download and extract LFW once; later calls are no-ops. Returns the image folder."""
    folder = data_dir / FOLDER_NAME
    if folder.is_dir() and any(folder.iterdir()):
        return folder
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / ARCHIVE_NAME
    if not archive.exists() or _sha256(archive) != SHA256:
        print(f"Downloading LFW (~230 MB) from {URL}")
        partial = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, partial)
        if _sha256(partial) != SHA256:
            partial.unlink()
            raise RuntimeError("LFW archive checksum mismatch; download aborted")
        partial.rename(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")
    archive.unlink()
    return folder


def identities(folder: Path, min_images: int, max_images: int | None = None) -> dict[str, list]:
    """Identity -> sorted image paths, for identities with ``min_images <= n (<= max)``."""
    out: dict[str, list[Path]] = {}
    for person in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(person.glob("*.jpg"))
        if len(files) >= min_images and (max_images is None or len(files) <= max_images):
            out[person.name] = files
    return out


def load_image(path: Path, crop: tuple[int, int, int, int] | None = CROP_BOX) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb.crop(crop) if crop else rgb)


def load_subset(
    folder: Path, min_images: int, cap: int | None = None, max_images: int | None = None
) -> FaceSet:
    """Load identities with at least ``min_images`` photos, keeping the first ``cap`` of each.

    Taking the first files in name order (not a random sample) keeps the subset identical on
    every machine without depending on a random generator.
    """
    groups = identities(folder, min_images, max_images)
    images, labels, names = [], [], []
    for label, (name, files) in enumerate(groups.items()):
        names.append(name)
        for path in files[:cap]:
            images.append(load_image(path))
            labels.append(label)
    return FaceSet(np.stack(images), np.asarray(labels), names)
