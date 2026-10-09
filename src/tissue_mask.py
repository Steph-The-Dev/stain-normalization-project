"""
Tissue Masking Utilities

Shared helpers to separate stained tissue from the bright glass background.
Used by all normalizers (Reinhard, Macenko, CUT) and the patch extractor.
"""

import cv2
import numpy as np
import numpy.typing as npt


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


def refine_mask(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Morphological open+close to remove mask noise and fill small holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
