"""Paths, constants and the shared visual identity.

Everything that a reader might want to change to re-run a variant of the study lives here,
so the numbers in the README can always be traced back to one place.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
SITE = ROOT / "site"
PORTRAIT = ROOT / "assets" / "portrait.jpg"

SEED = 42
IMG_SIZE = 224

# LFW protocol: identities with at least MIN_IMAGES images, capped at MAX_IMAGES images each,
# so that one prolific identity (George W. Bush has 530 photos) cannot dominate the accuracy.
LFW_MIN_IMAGES = 20
LFW_MAX_IMAGES = 20
# Identities with 10-19 images are disjoint from the evaluation set and are only used to fit
# the eigenfaces baseline, so the baseline never sees an evaluation identity.
LFW_BASELINE_MIN_IMAGES = 10
N_ENROLMENT_DRAWS = 20

# Visual identity shared across the portfolio.
INK = "#0f172a"
TEAL = "#0d9488"
AMBER = "#d97706"
SLATE = "#64748b"
GRID = "#e2e8f0"
