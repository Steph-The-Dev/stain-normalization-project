import pytest
import numpy as np

# Wir importieren jetzt BEIDE Methoden aus deinem Modul
from src.reinhard import (
    get_tissue_mask_hsv, 
    get_tissue_mask_manual,
    normalize_stain_reinhard_hsv_final, 
    normalize_stain_reinhard_custom
)

# --- 1. MASKEN TESTS ---

def test_hsv_mask_is_binary_and_correct_shape():
    """Prüft die HSV (Sättigungs) Maske."""
    dummy_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    mask = get_tissue_mask_hsv(dummy_img, saturation_threshold=15)
    
    assert np.all(np.isin(np.unique(mask), [0, 255])), "HSV-Maske ist nicht binär!"
    assert mask.shape == (100, 100), "HSV-Maske hat falsche Auflösung!"

def test_luma_mask_is_binary_and_correct_shape():
    """Prüft die Luma (Graustufen) Maske."""
    dummy_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    mask = get_tissue_mask_manual(dummy_img, threshold_value=210)
    
    assert np.all(np.isin(np.unique(mask), [0, 255])), "Luma-Maske ist nicht binär!"
    assert mask.shape == (100, 100), "Luma-Maske hat falsche Auflösung!"

# --- 2. PIPELINE TESTS (SHAPE & TYPE) ---

def test_normalization_hsv_preserves_shape_and_type():
    """Prüft den kompletten HSV-Render-Pfad."""
    src_dummy = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    trg_dummy = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    
    result = normalize_stain_reinhard_hsv_final(src_dummy, trg_dummy)
    
    assert result.shape == src_dummy.shape, "HSV-Render ändert Bildgröße!"
    assert result.dtype == np.uint8, "HSV-Render Output ist kein 8-Bit Bild!"

def test_normalization_luma_preserves_shape_and_type():
    """Prüft den kompletten Luma-Render-Pfad."""
    src_dummy = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    trg_dummy = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    
    result = normalize_stain_reinhard_custom(src_dummy, trg_dummy)
    
    assert result.shape == src_dummy.shape, "Luma-Render ändert Bildgröße!"
    assert result.dtype == np.uint8, "Luma-Render Output ist kein 8-Bit Bild!"

# --- 3. EDGE CASES (ZERO DIVISION) ---

def test_zero_division_protection_both_methods():
    """Prüft, ob komplett schwarze (leere) Bilder das System crashen."""
    black_src = np.zeros((50, 50, 3), dtype=np.uint8)
    black_trg = np.zeros((50, 50, 3), dtype=np.uint8)
    
    try:
        # Teste HSV
        res_hsv = normalize_stain_reinhard_hsv_final(black_src, black_trg)
        assert res_hsv is not None
        
        # Teste Luma
        res_luma = normalize_stain_reinhard_custom(black_src, black_trg)
        assert res_luma is not None
        
    except ZeroDivisionError:
        pytest.fail("Algorithmus ist bei schwarzen Bildern abgestürzt (Division durch Null)!")