"""
Very dark, nearly neutral pixels must not bias the Macenko stain vectors.

Background: in the Camelyon17 experiments (experiments/), 16 of 50 slides had a hematoxylin
vector pulled towards the grey axis by dark pixels. The HSV tissue mask does not remove them,
because HSV saturation is relative to brightness.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.normalizers.macenko import get_stain_matrix_macenko, rgb_to_od
from src.tissue_mask import get_tissue_mask_hsv

DEMO = Path(__file__).resolve().parents[1] / "data" / "raw" / "target.tif"
DARK_BGR = [(7, 3, 6), (35, 23, 37)]  # near-black; dark and barely coloured


def _stain_vectors(bgr, dark_threshold):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = get_tissue_mask_hsv(bgr, saturation_threshold=15)
    return get_stain_matrix_macenko(rgb_to_od(rgb), mask, dark_threshold=dark_threshold)


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _with_dark_pixels(bgr, colour, fraction=0.02, seed=0):
    out = bgr.copy()
    flat = out.reshape(-1, 3)
    idx = np.random.default_rng(seed).choice(len(flat), int(fraction * len(flat)), replace=False)
    flat[idx] = colour
    return out


@pytest.fixture(scope="module")
def demo_image():
    img = cv2.imread(str(DEMO))
    assert img is not None, f"demo image missing: {DEMO}"
    return img


@pytest.mark.parametrize("colour", DARK_BGR)
def test_dark_pixels_bias_vectors_without_filter(demo_image, colour):
    clean = _stain_vectors(demo_image, dark_threshold=None)
    dirty = _stain_vectors(_with_dark_pixels(demo_image, colour), dark_threshold=None)
    assert _angle_deg(clean[0], dirty[0]) > 3.0


@pytest.mark.parametrize("colour", DARK_BGR)
def test_dark_pixel_filter_keeps_vectors_stable(demo_image, colour):
    clean = _stain_vectors(demo_image, dark_threshold=60)
    dirty = _stain_vectors(_with_dark_pixels(demo_image, colour), dark_threshold=60)
    assert _angle_deg(clean[0], dirty[0]) < 1.0
    assert _angle_deg(clean[1], dirty[1]) < 1.0


def test_dark_pixel_filter_does_not_change_clean_image(demo_image):
    with_filter = _stain_vectors(demo_image, dark_threshold=60)
    without_filter = _stain_vectors(demo_image, dark_threshold=None)
    assert _angle_deg(with_filter[0], without_filter[0]) < 1.0
    assert _angle_deg(with_filter[1], without_filter[1]) < 1.0
