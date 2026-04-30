"""
Reinhard Stain Normalization Module

This module provides functions for color normalization of histological images 
using the Reinhard method with Luma and HSV-based tissue masking.
"""

import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional


def get_tissue_mask_luma(image: npt.NDArray[np.uint8], threshold_value: int = 210) -> npt.NDArray[np.uint8]:
    """
    Creates a tissue mask based on brightness (Luma-Key).

    Parameters
    ----------
    image : npt.NDArray[np.uint8]
        The source image in BGR format.
    threshold_value : int, optional
        The threshold for the Luma-Key, by default 210.

    Returns
    -------
    npt.NDArray[np.uint8]
        A binary mask where tissue is white (255) and background is black (0).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return mask


def get_tissue_mask_hsv(image: npt.NDArray[np.uint8], saturation_threshold: int = 15) -> npt.NDArray[np.uint8]:
    """
    Creates a tissue mask based on saturation (HSV Chroma-Key).

    Parameters
    ----------
    image : npt.NDArray[np.uint8]
        The source image in BGR format.
    saturation_threshold : int, optional
        The saturation threshold for the Chroma-Key, by default 15.

    Returns
    -------
    npt.NDArray[np.uint8]
        A binary mask where tissue is white (255) and background is black (0).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    _, mask = cv2.threshold(s_channel, saturation_threshold, 255, cv2.THRESH_BINARY)
    return mask


def get_mean_std_masked(image: npt.NDArray[np.float32], mask: Optional[npt.NDArray[np.uint8]] = None) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Calculates the mean and standard deviation for masked pixels.

    Parameters
    ----------
    image : npt.NDArray[np.float32]
        The image data (typically in LAB space).
    mask : npt.NDArray[np.uint8], optional
        Optional binary mask to isolate tissue, by default None.

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        A tuple containing (mean, std) for each channel.
    """
    mean, std = cv2.meanStdDev(image, mask=mask)
    return mean.flatten().astype(np.float32), std.flatten().astype(np.float32)


def _apply_reinhard_stats(
    src_lab: npt.NDArray[np.float32], 
    src_mean: npt.NDArray[np.float32], 
    src_std: npt.NDArray[np.float32], 
    target_mean: npt.NDArray[np.float32], 
    target_std: npt.NDArray[np.float32], 
    luma_blend: float = 0.0
) -> npt.NDArray[np.uint8]:
    """
    Internal helper to apply the Reinhard statistical transformation.
    """
    # Prevent division by zero
    src_std[src_std == 0] = 1e-5

    l, a, b = cv2.split(src_lab)
    
    # 1. Scale L-channel (Luminance)
    l_norm = (l - src_mean[0]) * (target_std[0] / src_std[0]) + target_mean[0]
    
    # Apply Luminance Blend (Opacity for micro-contrast preservation)
    l_final = (l_norm * (1.0 - luma_blend)) + (l * luma_blend)

    # 2. Scale A and B channels (Color)
    a_norm = (a - src_mean[1]) * (target_std[1] / src_std[1]) + target_mean[1]
    b_norm = (b - src_mean[2]) * (target_std[2] / src_std[2]) + target_mean[2]

    # Merge and clip to valid range
    result_lab = cv2.merge((l_final, a_norm, b_norm))
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def normalize_stain_reinhard_luma(
    src_img: npt.NDArray[np.uint8], 
    target_img: npt.NDArray[np.uint8], 
    src_thresh: int = 210, 
    target_thresh: int = 210, 
    luma_blend: float = 0.0
) -> npt.NDArray[np.uint8]:
    """
    Performs Reinhard normalization using Luma-masks for tissue isolation.

    Parameters
    ----------
    src_img : npt.NDArray[np.uint8]
        Source image to be normalized.
    target_img : npt.NDArray[np.uint8]
        Target reference image.
    src_thresh : int, optional
        Luma threshold for source tissue, by default 210.
    target_thresh : int, optional
        Luma threshold for target tissue, by default 210.
    luma_blend : float, optional
        Blending factor for original luminance (0.0 to 1.0), by default 0.0.

    Returns
    -------
    npt.NDArray[np.uint8]
        The normalized image in BGR format.
    """
    src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mask = get_tissue_mask_luma(src_img, threshold_value=src_thresh)
    target_mask = get_tissue_mask_luma(target_img, threshold_value=target_thresh)

    src_mean, src_std = get_mean_std_masked(src_lab, mask=src_mask)
    target_mean, target_std = get_mean_std_masked(target_lab, mask=target_mask)

    return _apply_reinhard_stats(src_lab, src_mean, src_std, target_mean, target_std, luma_blend)


def normalize_stain_reinhard_hsv(
    src_img: npt.NDArray[np.uint8], 
    target_img: npt.NDArray[np.uint8], 
    src_sat_thresh: int = 15, 
    target_sat_thresh: int = 15, 
    luma_blend: float = 0.0
) -> npt.NDArray[np.uint8]:
    """
    Performs Reinhard normalization using HSV-saturation masks (Recommended).

    Parameters
    ----------
    src_img : npt.NDArray[np.uint8]
        Source image to be normalized.
    target_img : npt.NDArray[np.uint8]
        Target reference image.
    src_sat_thresh : int, optional
        Saturation threshold for source tissue, by default 15.
    target_sat_thresh : int, optional
        Saturation threshold for target tissue, by default 15.
    luma_blend : float, optional
        Blending factor for original luminance (0.0 to 1.0), by default 0.0.

    Returns
    -------
    npt.NDArray[np.uint8]
        The normalized image in BGR format.
    """
    src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mask = get_tissue_mask_hsv(src_img, saturation_threshold=src_sat_thresh)
    target_mask = get_tissue_mask_hsv(target_img, saturation_threshold=target_sat_thresh)

    src_mean, src_std = get_mean_std_masked(src_lab, mask=src_mask)
    target_mean, target_std = get_mean_std_masked(target_lab, mask=target_mask)

    return _apply_reinhard_stats(src_lab, src_mean, src_std, target_mean, target_std, luma_blend)
