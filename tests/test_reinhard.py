import pytest
import numpy as np
from src.reinhard import get_tissue_mask_hsv, normalize_stain_reinhard_hsv_final

def test_hsv_mask_is_binary_and_correct_shape():
    """Prüft, ob die Maske wirklich nur Schwarz (0) und Weiß (255) enthält und 2D ist."""
    # 1. Arrange: Künstliches 100x100 RGB-Testbild (Rauschen) erstellen
    dummy_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # 2. Act: Maske generieren
    mask = get_tissue_mask_hsv(dummy_img, saturation_threshold=15)
    
    # 3. Assert: Behauptungen aufstellen (Wenn eine fehlschlägt, schlägt der Test fehl)
    unique_values = np.unique(mask)
    
    # Darf nur 0 oder 255 sein
    assert np.all(np.isin(unique_values, [0, 255])), "Maske enthält Werte außer 0 und 255!"
    # Muss exakt 100x100 Pixel groß sein (1 Kanal, Graustufe)
    assert mask.shape == (100, 100), "Maske hat die falsche Auflösung!"

def test_normalization_preserves_shape_and_type():
    """Prüft, ob das ausgegebene Bild die exakt gleiche Auflösung wie die Quelle hat."""
    # Source ist 50x50, Target ist 80x80 (unterschiedliche Größen sind in der Realität normal)
    src_dummy = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    trg_dummy = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    
    result = normalize_stain_reinhard_hsv_final(src_dummy, trg_dummy)
    
    # Das Ergebnis MUSS exakt die Auflösung der Quelle haben
    assert result.shape == src_dummy.shape, "Das normalisierte Bild hat seine Größe verändert!"
    # Das Ergebnis MUSS ein 8-Bit Bild sein
    assert result.dtype == np.uint8, "Das Ergebnis ist kein 8-Bit (uint8) Bild mehr!"

def test_zero_division_protection():
    """Prüft den Edge-Case: Was passiert, wenn man dem Algorithmus ein komplett schwarzes Bild gibt?"""
    # Komplett schwarze Bilder haben eine Standardabweichung von 0. 
    # Ohne unseren 1e-5 Schutz würde Python hier abstürzen (Division durch Null).
    black_src = np.zeros((50, 50, 3), dtype=np.uint8)
    black_trg = np.zeros((50, 50, 3), dtype=np.uint8)
    
    try:
        result = normalize_stain_reinhard_hsv_final(black_src, black_trg)
        assert result is not None
    except ZeroDivisionError:
        pytest.fail("Algorithmus ist abgestürzt wegen Division durch Null!")