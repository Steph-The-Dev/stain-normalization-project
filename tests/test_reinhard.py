import pytest
import numpy as np
from src.reinhard import (
    get_tissue_mask_hsv, 
    get_tissue_mask_luma,
    normalize_stain_reinhard_hsv, 
    normalize_stain_reinhard_luma
)

# --- 1. MASK TESTS ---

def test_hsv_mask_is_binary_and_correct_shape():
    """Verify that the HSV saturation mask is binary and maintains correct resolution."""
    dummy_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    mask = get_tissue_mask_hsv(dummy_img, saturation_threshold=15)
    
    assert np.all(np.isin(np.unique(mask), [0, 255])), "HSV mask is not binary!"
    assert mask.shape == (100, 100), "HSV mask has incorrect resolution!"

def test_luma_mask_is_binary_and_correct_shape():
    """Verify that the Luma grayscale mask is binary and maintains correct resolution."""
    dummy_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    mask = get_tissue_mask_luma(dummy_img, threshold_value=210)
    
    assert np.all(np.isin(np.unique(mask), [0, 255])), "Luma mask is not binary!"
    assert mask.shape == (100, 100), "Luma mask has incorrect resolution!"

# --- 2. PIPELINE TESTS (SHAPE & TYPE) ---

def test_normalization_hsv_preserves_shape_and_type():
    """Verify that the HSV normalization pipeline preserves image shape and data type."""
    src_dummy = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    trg_dummy = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    
    result = normalize_stain_reinhard_hsv(src_dummy, trg_dummy)
    
    assert result.shape == src_dummy.shape, "HSV normalization changed image size!"
    assert result.dtype == np.uint8, "HSV normalization output is not 8-bit!"

def test_normalization_luma_preserves_shape_and_type():
    """Verify that the Luma normalization pipeline preserves image shape and data type."""
    src_dummy = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    trg_dummy = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    
    result = normalize_stain_reinhard_luma(src_dummy, trg_dummy)
    
    assert result.shape == src_dummy.shape, "Luma normalization changed image size!"
    assert result.dtype == np.uint8, "Luma normalization output is not 8-bit!"

# --- 3. EDGE CASES (ZERO DIVISION) ---

def test_zero_division_protection_both_methods():
    """Ensure that completely black (empty) images do not cause the algorithm to crash."""
    black_src = np.zeros((50, 50, 3), dtype=np.uint8)
    black_trg = np.zeros((50, 50, 3), dtype=np.uint8)
    
    try:
        # Test HSV path
        res_hsv = normalize_stain_reinhard_hsv(black_src, black_trg)
        assert res_hsv is not None
        
        # Test Luma path
        res_luma = normalize_stain_reinhard_luma(black_src, black_trg)
        assert res_luma is not None
        
    except ZeroDivisionError:
        pytest.fail("Algorithm crashed on black images (ZeroDivisionError)!")
