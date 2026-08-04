"""
Tests for quantitative evaluation metrics module.
"""

import numpy as np
import pytest
from src.metrics import calculate_ssim, calculate_psnr, calculate_lab_color_distance, evaluate_normalization


def test_ssim_identical_images():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    ssim = calculate_ssim(img, img)
    assert pytest.approx(ssim, rel=1e-3) == 1.0


def test_psnr_identical_images():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    psnr = calculate_psnr(img, img)
    assert psnr == float("inf")


def test_lab_color_distance():
    img1 = np.full((50, 50, 3), 100, dtype=np.uint8)
    img2 = np.full((50, 50, 3), 150, dtype=np.uint8)
    dist = calculate_lab_color_distance(img1, img2)
    assert dist["delta_L"] > 0


def test_evaluate_normalization():
    src = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    norm = src.copy()
    trg = np.random.randint(80, 220, (100, 100, 3), dtype=np.uint8)

    results = evaluate_normalization(src, norm, trg)
    assert "ssim_structural_preservation" in results
    assert "psnr_db" in results
    assert "target_delta_L" in results
    assert "target_delta_ab" in results
    assert pytest.approx(results["ssim_structural_preservation"], rel=1e-3) == 1.0
