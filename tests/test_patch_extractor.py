"""
Tests for Tissue Patch Extractor module.
"""

import os
import numpy as np
import pytest
import cv2

from src.patch_extractor import (
    get_tissue_mask,
    calculate_tissue_ratio,
    TissuePatchExtractor,
)


def test_calculate_tissue_ratio():
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    assert calculate_tissue_ratio(empty_mask) == 0.0

    full_mask = np.full((100, 100), 255, dtype=np.uint8)
    assert calculate_tissue_ratio(full_mask) == 1.0

    half_mask = np.zeros((100, 100), dtype=np.uint8)
    half_mask[:50, :] = 255
    assert pytest.approx(calculate_tissue_ratio(half_mask)) == 0.5


def test_get_tissue_mask_invalid_method():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown tissue masking method"):
        get_tissue_mask(img, method="invalid")


def test_tissue_patch_extractor_invalid_params():
    with pytest.raises(ValueError, match="patch_size must be a positive integer"):
        TissuePatchExtractor(patch_size=0)

    with pytest.raises(ValueError, match="stride must be a positive integer"):
        TissuePatchExtractor(stride=-1)

    with pytest.raises(ValueError, match="min_tissue_ratio must be between"):
        TissuePatchExtractor(min_tissue_ratio=1.5)


def test_extract_patches_filtering():
    # Create a 512x512 image:
    # Top-Left 256x256: Hematoxylin purple tissue (R=120, G=40, B=160)
    # Bottom-Right 256x256: Glass background (R=245, G=245, B=245)
    img = np.full((512, 512, 3), 245, dtype=np.uint8)
    img[0:256, 0:256] = [120, 40, 160]

    extractor = TissuePatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.5)
    extracted = extractor.extract_patches(img)

    # Should keep only the top-left tissue patch, filtering out white background patches
    assert len(extracted) == 1
    patch, (y, x) = extracted[0]
    assert (y, x) == (0, 0)
    assert patch.shape == (256, 256, 3)


def test_extract_and_save_patches(tmp_path):
    img = np.full((512, 512, 3), 245, dtype=np.uint8)
    img[0:256, 0:256] = [120, 40, 160]

    out_dir = str(tmp_path / "patches")
    extractor = TissuePatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.5)
    saved_paths = extractor.extract_and_save_patches(img, output_dir=out_dir, prefix="slide1")

    assert len(saved_paths) == 1
    assert os.path.exists(saved_paths[0])
    saved_img = cv2.imread(saved_paths[0])
    assert saved_img.shape == (256, 256, 3)
