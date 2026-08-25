"""
Reinhard Stain Normalizer implementation using the Strategy Pattern.
"""

from typing import Optional
import cv2
import numpy as np
import numpy.typing as npt

from src.normalizers.base import BaseStainNormalizer
from src.reinhard import (
    get_tissue_mask_hsv,
    get_tissue_mask_luma,
    get_mean_std_masked,
    _apply_reinhard_stats,
)


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

    def fit(self, target_image: npt.NDArray[np.uint8]) -> "ReinhardNormalizer":
        """
        Extract target mean and std in CIELAB space.
        """
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        if self.mask_method == "luma":
            mask = get_tissue_mask_luma(target_image, threshold_value=self.threshold)
        else:
            mask = get_tissue_mask_hsv(target_image, saturation_threshold=self.threshold)

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
        if self.mask_method == "luma":
            mask = get_tissue_mask_luma(source_image, threshold_value=self.threshold)
        else:
            mask = get_tissue_mask_hsv(source_image, saturation_threshold=self.threshold)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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
