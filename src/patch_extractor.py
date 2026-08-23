"""
Tissue Patch Extraction & Background Filtering Module.

Extracts patches from whole slide images/slide crops while filtering out
non-tissue background areas (glass slides) using HSV/Luma tissue masking.
"""

import os
from typing import List, Tuple, Generator, Optional
import cv2
import numpy as np
import numpy.typing as npt

from src.reinhard import get_tissue_mask_hsv, get_tissue_mask_luma


def get_tissue_mask(img: npt.NDArray[np.uint8], method: str = "hsv") -> npt.NDArray[np.uint8]:
    """
    Computes a binary tissue mask (255 for tissue, 0 for background).
    """
    if method.lower() == "hsv":
        return get_tissue_mask_hsv(img)
    elif method.lower() == "luma":
        return get_tissue_mask_luma(img)
    else:
        raise ValueError(f"Unknown tissue masking method: '{method}'. Supported: 'hsv', 'luma'.")


def calculate_tissue_ratio(mask_patch: npt.NDArray[np.uint8]) -> float:
    """
    Calculates the proportion of tissue pixels in a binary mask patch.
    """
    if mask_patch.size == 0:
        return 0.0
    return float(np.mean(mask_patch > 0))


class TissuePatchExtractor:
    """
    Grid-based patch extractor with background filtering for histological slides.
    """

    def __init__(
        self,
        patch_size: int = 256,
        stride: int = 256,
        min_tissue_ratio: float = 0.5,
        mask_method: str = "hsv",
    ) -> None:
        """
        Args:
            patch_size: Square patch size in pixels.
            stride: Step size for grid sliding. Defaults to patch_size (non-overlapping).
            min_tissue_ratio: Minimum fraction of tissue pixels required [0.0 - 1.0].
            mask_method: 'hsv' or 'luma' for tissue detection.
        """
        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")
        if not (0.0 <= min_tissue_ratio <= 1.0):
            raise ValueError("min_tissue_ratio must be between 0.0 and 1.0.")

        self.patch_size = patch_size
        self.stride = stride
        self.min_tissue_ratio = min_tissue_ratio
        self.mask_method = mask_method

    def extract_patches(
        self, image: npt.NDArray[np.uint8]
    ) -> List[Tuple[npt.NDArray[np.uint8], Tuple[int, int]]]:
        """
        Extracts valid tissue patches from an RGB image.

        Args:
            image: Input RGB image of shape (H, W, 3).

        Returns:
            List of tuples: (patch_rgb, (top_left_y, top_left_x))
        """
        h, w = image.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            return []

        tissue_mask = get_tissue_mask(image, method=self.mask_method)
        patches = []

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y : y + self.patch_size, x : x + self.patch_size]
                mask_patch = tissue_mask[y : y + self.patch_size, x : x + self.patch_size]

                ratio = calculate_tissue_ratio(mask_patch)
                if ratio >= self.min_tissue_ratio:
                    patches.append((patch, (y, x)))

        return patches

    def extract_and_save_patches(
        self,
        image: npt.NDArray[np.uint8],
        output_dir: str,
        prefix: str = "patch",
    ) -> List[str]:
        """
        Extracts tissue patches and saves them as PNG files in output_dir.

        Returns:
            List of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        extracted = self.extract_patches(image)
        saved_paths = []

        for patch, (y, x) in extracted:
            filename = f"{prefix}_y{y}_x{x}.png"
            filepath = os.path.join(output_dir, filename)
            # Write as BGR for OpenCV
            cv2.imwrite(filepath, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            saved_paths.append(filepath)

        return saved_paths
