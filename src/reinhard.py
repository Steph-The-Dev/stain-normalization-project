"""
Reinhard Stain Normalization Module
Beinhaltet Funktionen zur Farb-Normalisierung histologischer Bilder 
mittels Luma- und HSV-basiertem Tissue-Masking.
"""

import cv2
import numpy as np

def get_tissue_mask_manual(image, threshold_value=210):
    """Erstellt eine Gewebe-Maske basierend auf der Helligkeit (Luma-Key)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return mask

def get_tissue_mask_hsv(image, saturation_threshold=15):
    """Erstellt eine Gewebe-Maske basierend auf der Sättigung (HSV Chroma-Key)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    _, mask = cv2.threshold(s_channel, saturation_threshold, 255, cv2.THRESH_BINARY)
    return mask

def get_mean_std_masked(image, mask=None):
    """Berechnet Mittelwert und Standardabweichung für maskierte Pixel."""
    mean, std = cv2.meanStdDev(image, mask=mask)
    return mean.flatten(), std.flatten()

def normalize_stain_reinhard_custom(src_img, target_img, src_thresh=210, target_thresh=210):
    """Reinhard-Normalisierung mit Luma-Masken."""
    src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mask = get_tissue_mask_manual(src_img, threshold_value=src_thresh)
    target_mask = get_tissue_mask_manual(target_img, threshold_value=target_thresh)

    src_mean, src_std = get_mean_std_masked(src_lab, mask=src_mask)
    target_mean, target_std = get_mean_std_masked(target_lab, mask=target_mask)

    src_std[src_std == 0] = 1e-5

    l, a, b = cv2.split(src_lab)
    l_norm = (l - src_mean[0]) * (target_std[0] / src_std[0]) + target_mean[0]
    a_norm = (a - src_mean[1]) * (target_std[1] / src_std[1]) + target_mean[1]
    b_norm = (b - src_mean[2]) * (target_std[2] / src_std[2]) + target_mean[2]

    result_lab = cv2.merge((l_norm, a_norm, b_norm))
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

def normalize_stain_reinhard_hsv_final(src_img, target_img, src_sat_thresh=15, target_sat_thresh=15):
    """Reinhard-Normalisierung mit HSV-Sättigungs-Masken (Empfohlen)."""
    src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mask = get_tissue_mask_hsv(src_img, saturation_threshold=src_sat_thresh)
    target_mask = get_tissue_mask_hsv(target_img, saturation_threshold=target_sat_thresh)

    src_mean, src_std = get_mean_std_masked(src_lab, mask=src_mask)
    target_mean, target_std = get_mean_std_masked(target_lab, mask=target_mask)

    src_std[src_std == 0] = 1e-5

    l, a, b = cv2.split(src_lab)
    l_norm = (l - src_mean[0]) * (target_std[0] / src_std[0]) + target_mean[0]
    a_norm = (a - src_mean[1]) * (target_std[1] / src_std[1]) + target_mean[1]
    b_norm = (b - src_mean[2]) * (target_std[2] / src_std[2]) + target_mean[2]

    result_lab = cv2.merge((l_norm, a_norm, b_norm))
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)