"""
Quantitative Evaluation Metrics for Image Quality & Stain Normalization.

This module provides metrics to evaluate structural preservation (SSIM),
signal fidelity (PSNR), and perceptual color shift in digital pathology images.
"""

import cv2
import numpy as np
import numpy.typing as npt
from typing import Dict, Union


def calculate_ssim(
    img1: npt.NDArray[np.uint8], 
    img2: npt.NDArray[np.uint8],
    win_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03
) -> float:
    """
    Calculates the Mean Structural Similarity Index (SSIM) between two images.

    Parameters
    ----------
    img1 : npt.NDArray[np.uint8]
        First image (e.g. source image).
    img2 : npt.NDArray[np.uint8]
        Second image (e.g. normalized image).
    win_size : int, optional
        Gaussian window size, by default 11.
    k1 : float, optional
        Algorithm constant, by default 0.01.
    k2 : float, optional
        Algorithm constant, by default 0.03.

    Returns
    -------
    float
        SSIM value between -1.0 and 1.0 (1.0 indicates perfect structural identity).
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    # Convert to grayscale float32 if multichannel
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g1 = img1.astype(np.float64)
        g2 = img2.astype(np.float64)

    c1 = (k1 * 255.0) ** 2
    c2 = (k2 * 255.0) ** 2

    # Gaussian kernel
    kernel = cv2.getGaussianKernel(win_size, 1.5)
    window = np.outer(kernel, kernel.T)

    mu1 = cv2.filter2D(g1, -1, window, borderType=cv2.BORDER_REPLICATE)
    mu2 = cv2.filter2D(g2, -1, window, borderType=cv2.BORDER_REPLICATE)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(g1 ** 2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_sq
    sigma2_sq = cv2.filter2D(g2 ** 2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu2_sq
    sigma12 = cv2.filter2D(g1 * g2, -1, window, borderType=cv2.BORDER_REPLICATE) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(np.mean(ssim_map))


def calculate_psnr(
    img1: npt.NDArray[np.uint8], 
    img2: npt.NDArray[np.uint8]
) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in dB.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def calculate_lab_color_distance(
    src_img: npt.NDArray[np.uint8], 
    target_img: npt.NDArray[np.uint8]
) -> Dict[str, float]:
    """
    Calculates color distribution differences in CIELAB color space.

    Returns
    -------
    Dict[str, float]
        Dictionary with delta_L (luminance difference) and delta_ab (chromaticity difference).
    """
    lab_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_trg = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    mean_src = np.mean(lab_src, axis=(0, 1))
    mean_trg = np.mean(lab_trg, axis=(0, 1))

    delta_l = float(abs(mean_src[0] - mean_trg[0]))
    delta_ab = float(np.linalg.norm(mean_src[1:] - mean_trg[1:]))

    return {
        "delta_L": delta_l,
        "delta_ab": delta_ab,
    }


def evaluate_normalization(
    source: npt.NDArray[np.uint8],
    normalized: npt.NDArray[np.uint8],
    target: npt.NDArray[np.uint8]
) -> Dict[str, float]:
    """
    Runs full quantitative benchmark suite on a normalization operation.

    Parameters
    ----------
    source : npt.NDArray[np.uint8]
        Original un-normalized source image.
    normalized : npt.NDArray[np.uint8]
        Transformed normalized image.
    target : npt.NDArray[np.uint8]
        Target reference standard.

    Returns
    -------
    Dict[str, float]
        Comprehensive dictionary of structural preservation & color alignment metrics.
    """
    # 1. Structural Preservation (Source vs Normalized)
    ssim_value = calculate_ssim(source, normalized)
    psnr_value = calculate_psnr(source, normalized)

    # 2. Color Alignment (Normalized vs Target)
    color_metrics = calculate_lab_color_distance(normalized, target)

    return {
        "ssim_structural_preservation": ssim_value,
        "psnr_db": psnr_value,
        "target_delta_L": color_metrics["delta_L"],
        "target_delta_ab": color_metrics["delta_ab"],
    }
