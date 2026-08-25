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
    img_float = np.clip(img.astype(np.float64) / io, 1e-5, 1.0)
    return -np.log10(img_float)


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

    # Filter low OD background pixels
    od_tissue = od_tissue[np.all(od_tissue > beta, axis=1)]

    # Standard clinical H&E reference stain vectors (Hematoxylin & Eosin)
    default_stain_matrix = np.array([
        [0.65, 0.70, 0.29],  # Hematoxylin
        [0.07, 0.99, 0.11],  # Eosin
    ], dtype=np.float64)
    default_stain_matrix /= np.linalg.norm(default_stain_matrix, axis=1, keepdims=True)

    if len(od_tissue) < 50:
        return default_stain_matrix

    # Compute SVD on tissue OD covariance plane
    _, _, vh = np.linalg.svd(od_tissue, full_matrices=False)
    V = vh[:2].copy()

    # Enforce positive orientation of SVD basis vectors to prevent collinear projection artifact
    if V[0, 0] < 0:
        V[0] = -V[0]
    if V[1, 0] < 0:
        V[1] = -V[1]

    # Project OD data onto 2D principal plane
    proj = np.dot(od_tissue, V.T)

    # Calculate angles on 2D plane
    angles = np.arctan2(proj[:, 1], proj[:, 0])

    # Find robust extreme percentiles (alpha and 100-alpha)
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)

    v1 = np.dot(V.T, np.array([np.cos(min_angle), np.sin(min_angle)]))
    v2 = np.dot(V.T, np.array([np.cos(max_angle), np.sin(max_angle)]))

    # Ensure positive OD space directions
    if v1[0] < 0:
        v1 = -v1
    if v2[0] < 0:
        v2 = -v2

    # Check angular separation (if vectors are nearly collinear, use fallback)
    cosine_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
    if cosine_sim > 0.99 or np.isnan(cosine_sim):
        return default_stain_matrix

    # Order vectors: Hematoxylin has higher Red absorption (channel 0)
    if v1[0] > v2[0]:
        stain_matrix = np.array([v1, v2])
    else:
        stain_matrix = np.array([v2, v1])

    # Normalize vectors to unit length
    stain_matrix /= np.linalg.norm(stain_matrix, axis=1, keepdims=True)
    return stain_matrix


class MacenkoNormalizer(BaseStainNormalizer):
    """
    Macenko Stain Normalizer using SVD on Optical Density space.
    """

    def __init__(self, saturation_threshold: int = 15, beta: float = 0.15) -> None:
        super().__init__()
        self.saturation_threshold = saturation_threshold
        self.beta = beta
        self.target_stain_matrix: Optional[npt.NDArray[np.float64]] = None
        self.target_max_concentrations: Optional[npt.NDArray[np.float64]] = None

    def fit(self, target_image: npt.NDArray[np.uint8]) -> "MacenkoNormalizer":
        target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        target_od = rgb_to_od(target_rgb)
        mask = get_tissue_mask_hsv(target_image, saturation_threshold=self.saturation_threshold)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.target_stain_matrix = get_stain_matrix_macenko(target_od, mask, beta=self.beta)

        target_od_flat = target_od.reshape(-1, 3)
        target_concentrations, _, _, _ = np.linalg.lstsq(
            self.target_stain_matrix.T, target_od_flat.T, rcond=None
        )
        target_concentrations = np.maximum(target_concentrations, 0)

        self.target_max_concentrations = np.percentile(target_concentrations, 99, axis=1, keepdims=True)
        self.target_max_concentrations[self.target_max_concentrations == 0] = 1e-5
        self.is_fitted = True
        return self

    def transform(self, source_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if not self.is_fitted or self.target_stain_matrix is None or self.target_max_concentrations is None:
            raise RuntimeError("MacenkoNormalizer must be fit before transform().")

        source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        source_od = rgb_to_od(source_rgb)
        mask = get_tissue_mask_hsv(source_image, saturation_threshold=self.saturation_threshold)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        source_stain_matrix = get_stain_matrix_macenko(source_od, mask, beta=self.beta)

        h, w, _ = source_rgb.shape
        source_od_flat = source_od.reshape(-1, 3)

        source_concentrations, _, _, _ = np.linalg.lstsq(
            source_stain_matrix.T, source_od_flat.T, rcond=None
        )
        source_concentrations = np.maximum(source_concentrations, 0)

        source_max_concentrations = np.percentile(source_concentrations, 99, axis=1, keepdims=True)
        source_max_concentrations[source_max_concentrations == 0] = 1e-5

        normalized_concentrations = source_concentrations * (
            self.target_max_concentrations / source_max_concentrations
        )

        normalized_od_flat = np.dot(self.target_stain_matrix.T, normalized_concentrations).T
        normalized_od = normalized_od_flat.reshape(h, w, 3)

        normalized_rgb = od_to_rgb(normalized_od)
        normalized_bgr = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)

        # Preserve original background pixels using tissue mask
        mask_f = (mask > 0).astype(np.float32)[:, :, None]
        result = normalized_bgr.astype(np.float32) * mask_f + source_image.astype(np.float32) * (1.0 - mask_f)
        return np.clip(result, 0, 255).astype(np.uint8)
