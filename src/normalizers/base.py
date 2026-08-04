"""
Abstract Base Class for Stain Normalization Algorithms.

This module enforces the Strategy Pattern across all stain normalizers
(classical statistical methods, matrix decompositions, and neural networks).
"""

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class BaseStainNormalizer(ABC):
    """
    Abstract base class for all stain normalization implementations.
    """

    def __init__(self) -> None:
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, target_image: npt.NDArray[np.uint8]) -> "BaseStainNormalizer":
        """
        Extracts target stain statistics / features from a target image standard.

        Parameters
        ----------
        target_image : npt.NDArray[np.uint8]
            Target reference image in BGR format.

        Returns
        -------
        BaseStainNormalizer
            Self instance (for method chaining).
        """
        pass

    @abstractmethod
    def transform(self, source_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Normalizes the source image to match the target stain distribution.

        Parameters
        ----------
        source_image : npt.NDArray[np.uint8]
            Source image in BGR format.

        Returns
        -------
        npt.NDArray[np.uint8]
            Normalized image in BGR format.
        """
        pass

    def fit_transform(
        self, 
        source_image: npt.NDArray[np.uint8], 
        target_image: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """
        Convenience method to fit target standard and transform source image in one step.
        """
        return self.fit(target_image).transform(source_image)
