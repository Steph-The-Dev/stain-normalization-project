"""
Reinhard Stain Normalization

Color normalization of histological images using the Reinhard method in
CIELAB space with Luma- or HSV-based tissue masking.

Provides two interfaces on top of the same math:

- ``ReinhardNormalizer``: fit/transform class following the shared
  ``BaseStainNormalizer`` strategy interface (used by the app's full-res
  export and interchangeable with Macenko/CUT).
- ``normalize_stain_reinhard_hsv`` / ``normalize_stain_reinhard_luma``:
  one-shot convenience functions (used for fast previews, batch and video
  processing).
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from src.normalizers.base import BaseStainNormalizer
from src.tissue_mask import get_tissue_mask_hsv, get_tissue_mask_luma, refine_mask


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
    luma_blend: float = 0.0,
    mask: Optional[npt.NDArray[np.uint8]] = None
) -> npt.NDArray[np.uint8]:
    """
    Internal helper to apply the Reinhard statistical transformation.
    """
    # Prevent division by zero
    src_std = src_std.copy()
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
    normalized_lab = cv2.merge((l_final, a_norm, b_norm))
    normalized_lab = np.clip(normalized_lab, 0, 255)

    if mask is not None:
        # Alpha blend: only transform tissue pixels, preserve original background
        mask_f = (mask > 0).astype(np.float32)[:, :, None]
        result_lab = normalized_lab * mask_f + src_lab * (1.0 - mask_f)
    else:
        result_lab = normalized_lab

    return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


class ReinhardNormalizer(BaseStainNormalizer):
    """
    Reinhard Color Normalizer operating in CIELAB color space with tissue masking.
    """

    def __init__(
        self,
        mask_method: str = "hsv",
        threshold: int = 15,
        luma_blend: float = 0.0,
    ) -> None:
        super().__init__()
        self.mask_method = mask_method
        self.threshold = threshold
        self.luma_blend = luma_blend

        self.target_mean: Optional[npt.NDArray[np.float32]] = None
        self.target_std: Optional[npt.NDArray[np.float32]] = None

    def _mask(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if self.mask_method == "luma":
            mask = get_tissue_mask_luma(image, threshold_value=self.threshold)
        else:
            mask = get_tissue_mask_hsv(image, saturation_threshold=self.threshold)
        return refine_mask(mask)

    def fit(self, target_image: npt.NDArray[np.uint8]) -> "ReinhardNormalizer":
        """
        Extract target mean and std in CIELAB space.
        """
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        mask = self._mask(target_image)

        self.target_mean, self.target_std = get_mean_std_masked(target_lab, mask=mask)
        self.is_fitted = True
        return self

    def transform(self, source_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Normalize source image using stored target stats.
        """
        if not self.is_fitted or self.target_mean is None or self.target_std is None:
            raise RuntimeError("ReinhardNormalizer must be fit before calling transform().")

        src_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        mask = self._mask(source_image)

        src_mean, src_std = get_mean_std_masked(src_lab, mask=mask)
        return _apply_reinhard_stats(
            src_lab,
            src_mean,
            src_std,
            self.target_mean,
            self.target_std,
            luma_blend=self.luma_blend,
            mask=mask,
        )


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

    src_mask = refine_mask(get_tissue_mask_luma(src_img, threshold_value=src_thresh))
    target_mask = refine_mask(get_tissue_mask_luma(target_img, threshold_value=target_thresh))

    src_mean, src_std = get_mean_std_masked(src_lab, mask=src_mask)
    target_mean, target_std = get_mean_std_masked(target_lab, mask=target_mask)

    return _apply_reinhard_stats(src_lab, src_mean, src_std, target_mean, target_std, luma_blend, mask=src_mask)


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

    src_mask = refine_mask(get_tissue_mask_hsv(src_img, saturation_threshold=src_sat_thresh))
    target_mask = refine_mask(get_tissue_mask_hsv(target_img, saturation_threshold=target_sat_thresh))

    src_mean, src_std = get_mean_std_masked(src_lab, mask=src_mask)
    target_mean, target_std = get_mean_std_masked(target_lab, mask=target_mask)

    return _apply_reinhard_stats(src_lab, src_mean, src_std, target_mean, target_std, luma_blend, mask=src_mask)
