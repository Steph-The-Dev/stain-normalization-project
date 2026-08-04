"""
Tests for Stain Normalizers abstraction layer.
"""

import numpy as np
import pytest
from src.normalizers import BaseStainNormalizer, ReinhardNormalizer, MacenkoNormalizer


def test_base_normalizer_not_fitted_by_default():
    normalizer = ReinhardNormalizer()
    assert not normalizer.is_fitted
    
    src = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="must be fit"):
        normalizer.transform(src)


def test_reinhard_normalizer_fit_transform():
    np.random.seed(42)
    src = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    target = np.random.randint(80, 220, (100, 100, 3), dtype=np.uint8)

    normalizer = ReinhardNormalizer(mask_method="hsv")
    result = normalizer.fit_transform(src, target)

    assert normalizer.is_fitted
    assert result.shape == src.shape
    assert result.dtype == np.uint8


def test_macenko_normalizer_fit_transform():
    np.random.seed(42)
    src = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    target = np.random.randint(80, 220, (100, 100, 3), dtype=np.uint8)

    normalizer = MacenkoNormalizer()
    result = normalizer.fit_transform(src, target)

    assert normalizer.is_fitted
    assert result.shape == src.shape
    assert result.dtype == np.uint8

