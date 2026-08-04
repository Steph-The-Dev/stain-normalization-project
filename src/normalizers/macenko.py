"""
Macenko Stain Normalizer implementation using Optical Density & SVD.
"""

from typing import Optional
import cv2
import numpy as np
import numpy.typing as npt

from src.normalizers.base import BaseStainNormalizer
from src.reinhard import get_tissue_mask_hsv


def rgb_to_od(img: npt.NDArray[np.uint8], io: float = 255.0) -> npt.NDArray[np.float64]:
    """
    Converts RGB image to Optical Density (OD) space using Beer-Lambert law:
    OD = -log10(I / I0)
    """
    img_float = img.astype(np.float64)
    img_float[img_float == 0] = 1.0  # Prevent log(0)
    return -np.log10(img_float / io)


def od_to_rgb(od: npt.NDArray[np.float64], io: float = 255.0) -> npt.NDArray[np.uint8]:
    """
    Converts Optical Density (OD) space back to RGB space:
    I = I0 * 10^(-OD)
    """
    rgb = io * (10.0 ** (-od))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def get_stain_matrix_macenko(
    od: npt.NDArray[np.float64], 
    mask: npt.NDArray[np.uint8],
    beta: float = 0.15,
    alpha: float = 1.0
) -> npt.NDArray[np.float64]:
    """
    Extracts 2x3 Stain Matrix (Hematoxylin and Eosin vectors) using SVD and robust percentile projection.
    """
    od_flat = od.reshape(-1, 3)
    mask_flat = mask.reshape(-1) > 0
    od_tissue = od_flat[mask_flat]

    # Filter low OD pixels
    od_tissue = od_tissue[np.all(od_tissue > beta, axis=1)]

    if len(od_tissue) < 10:
        # Fallback to default H&E stain vectors if tissue signal is weak
        return np.array([
            [0.65, 0.70, 0.29],  # Hematoxylin
            [0.07, 0.99, 0.11],  # Eosin
        ], dtype=np.float64)

    # Compute SVD on covariance plane
    _, _, vh = np.linalg.svd(od_tissue, full_matrices=False)
    # Project data onto plane spanned by first two eigenvectors
    plane = vh[:2]
    proj = np.dot(od_tissue, plane.T)

    # Calculate angles on the 2D plane
    angles = np.arctan2(proj[:, 1], proj[:, 0])

    # Find robust extreme percentiles (alpha and 100-alpha)
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)

    v_min = np.dot(plane.T, np.array([np.cos(min_angle), np.sin(min_angle)]))
    v_max = np.dot(plane.T, np.array([np.cos(max_angle), np.sin(max_angle)]))

    # Order vectors (Hematoxylin has higher OD in first channel)
    if v_min[0] > v_max[0]:
        stain_matrix = np.array([v_min, v_max])
    else:
        stain_matrix = np.array([v_max, v_min])

    # Normalize vectors to unit length
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)
    return stain_matrix


class MacenkoNormalizer(BaseStainNormalizer):
    """
    Macenko Stain Normalizer using SVD on Optical Density space.
    """

    def __init__(self, saturation_threshold: int = 15) -> None:
        super().__init__()
        self.saturation_threshold = saturation_threshold
        self.target_stain_matrix: Optional[npt.NDArray[np.float64]] = None
        self.target_max_concentrations: Optional[npt.NDArray[np.float64]] = None

    def fit(self, target_image: npt.NDArray[np.uint8]) -> "MacenkoNormalizer":
        target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        target_od = rgb_to_od(target_rgb)
        mask = get_tissue_mask_hsv(target_image, saturation_threshold=self.saturation_threshold)

        self.target_stain_matrix = get_stain_matrix_macenko(target_od, mask)

        target_od_flat = target_od.reshape(-1, 3)
        target_concentrations, _, _, _ = np.linalg.lstsq(
            self.target_stain_matrix.T, target_od_flat.T, rcond=None
        )
        self.target_max_concentrations = np.percentile(target_concentrations, 99, axis=1, keepdims=True)
        self.is_fitted = True
        return self

    def transform(self, source_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if not self.is_fitted or self.target_stain_matrix is None or self.target_max_concentrations is None:
            raise RuntimeError("MacenkoNormalizer must be fit before transform().")

        source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        source_od = rgb_to_od(source_rgb)
        mask = get_tissue_mask_hsv(source_image, saturation_threshold=self.saturation_threshold)

        source_stain_matrix = get_stain_matrix_macenko(source_od, mask)

        h, w, _ = source_rgb.shape
        source_od_flat = source_od.reshape(-1, 3)

        source_concentrations, _, _, _ = np.linalg.lstsq(
            source_stain_matrix.T, source_od_flat.T, rcond=None
        )
        source_max_concentrations = np.percentile(source_concentrations, 99, axis=1, keepdims=True)
        source_max_concentrations[source_max_concentrations == 0] = 1e-5

        normalized_concentrations = source_concentrations * (
            self.target_max_concentrations / source_max_concentrations
        )

        normalized_od_flat = np.dot(self.target_stain_matrix.T, normalized_concentrations).T
        normalized_od = normalized_od_flat.reshape(h, w, 3)

        normalized_rgb = od_to_rgb(normalized_od)
        return cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)
