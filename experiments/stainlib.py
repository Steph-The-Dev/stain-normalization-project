"""
stainlib – Farb- und Shift-Werkzeuge für die Stain-Normalization-Experimente.

Enthält nur NumPy/scikit-image/scikit-learn-Code (kein PyTorch):
  * Macenko-Normalisierung mit getrenntem Fit (pro Slide) und Anwendung (pro Patch)
  * HED-Farbaugmentation und kontrollierte HED-Verschiebungen für Robustheitstests
  * Farbstatistiken und Shift-Metriken (MMD, Fréchet-Distanz, Herkunfts-Vorhersagbarkeit)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from skimage.color import hed2rgb, rgb2hed, rgb2lab

# --------------------------------------------------------------------------------------
# Macenko
# --------------------------------------------------------------------------------------

# Referenzwerte aus Macenkos Originalcode (auch in torchstain). Werden im Experiment
# durch eine Referenz ersetzt, die aus den Trainingsdaten geschätzt wird.
HE_REF_DEFAULT = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
MAXC_REF_DEFAULT = np.array([1.9705, 1.0308])


@dataclass
class StainParams:
    """Geschätzte Stain-Matrix (Spalten: H, E) und 99.-Perzentil der Konzentrationen."""

    HE: np.ndarray  # 3 x 2
    maxC: np.ndarray  # 2

    def angles_deg(self) -> np.ndarray:
        """Winkel der H- und E-Vektoren zur Diagonalen (1,1,1)/sqrt(3) in Grad – kompakte Beschreibung für Plots."""
        d = np.ones(3) / np.sqrt(3)
        cos = np.clip(self.HE.T @ d, -1, 1)
        return np.degrees(np.arccos(cos))


REFERENCE_DEFAULT = StainParams(HE_REF_DEFAULT, MAXC_REF_DEFAULT)


def _to_od(img: np.ndarray, Io: float) -> np.ndarray:
    I = img.reshape(-1, 3).astype(np.float64)
    return -np.log((I + 1.0) / Io)


def macenko_fit(img: np.ndarray, Io: float = 240, alpha: float = 1, beta: float = 0.15,
                min_pixels: int = 200, min_angle_deg: float = 5.0,
                dark_thresh: float | None = None) -> StainParams:
    """Schätzt Stain-Matrix und Intensität aus einem Bild (oder einem Mosaik vieler Patches).

    img: uint8-RGB-Array beliebiger Form (..., 3).
    dark_thresh: Pixel, deren hellster Kanal unter diesem Wert liegt (fast schwarz: Scan-Rand, Tinte,
        Falten), werden für die Schätzung ignoriert. None = Original-Macenko ohne diesen Filter.
        Hintergrund: Schwarz liegt im OD-Raum auf der Graudiagonale (OD ≈ ln 240 in allen Kanälen)
        und zieht den Hämatoxylin-Vektor dorthin; die H-Intensität landet dann bei √3·ln 240 ≈ 9,49.
    Wirft ValueError, wenn zu wenig Gewebepixel vorhanden sind oder die beiden
    Farbvektoren weniger als min_angle_deg auseinanderliegen.
    """
    OD = _to_od(img, Io)
    if dark_thresh is not None:
        OD = OD[img.reshape(-1, 3).max(axis=1) >= dark_thresh]
    ODhat = OD[~np.any(OD < beta, axis=1)]
    if ODhat.shape[0] < min_pixels:
        raise ValueError(f"zu wenig Gewebepixel ({ODhat.shape[0]})")

    _, eig = np.linalg.eigh(np.cov(ODhat.T))
    plane = eig[:, 1:3].copy()  # die zwei grössten Eigenvektoren (eigh sortiert aufsteigend)
    # Vorzeichen festlegen, damit der Pixelfächer um phi ~ 0 liegt (kein Sprung bei ±pi)
    for j in range(2):
        if plane[0, j] < 0:
            plane[:, j] *= -1

    t = ODhat @ plane
    phi = np.arctan2(t[:, 1], t[:, 0])
    lo, hi = np.percentile(phi, [alpha, 100 - alpha])
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])

    # Hämatoxylin absorbiert Rot stärker -> grössere erste Komponente
    HE = np.stack([v1, v2], axis=1) if v1[0] > v2[0] else np.stack([v2, v1], axis=1)
    HE = HE / np.linalg.norm(HE, axis=0, keepdims=True)

    # Plausibilität: fast parallele Vektoren heissen, dass nur ein Farbstoff erfasst wurde
    # (z. B. reines Stroma, oder Eosin-Pixel vom Hintergrundfilter entfernt)
    if HE[:, 0] @ HE[:, 1] > np.cos(np.radians(min_angle_deg)):
        raise ValueError("Farbvektoren nicht trennbar (nur ein Farbstoff erkannt)")

    C = np.linalg.pinv(HE) @ OD.T
    maxC = np.percentile(C, 99, axis=1)
    if np.any(maxC <= 1e-6):
        raise ValueError("degenerierte Konzentrationen")
    return StainParams(HE, maxC)


def macenko_apply(img: np.ndarray, src: StainParams, ref: StainParams = REFERENCE_DEFAULT,
                  Io: float = 240) -> np.ndarray:
    """Normalisiert ein Bild mit bereits geschätzten Quellparametern auf die Referenz."""
    shape = img.shape
    OD = _to_od(img, Io)
    C = np.linalg.pinv(src.HE) @ OD.T
    C *= (ref.maxC / src.maxC)[:, None]
    out = Io * np.exp(-ref.HE @ C)
    return np.clip(out, 0, 255).T.reshape(shape).astype(np.uint8)


def macenko_normalize(img: np.ndarray, ref: StainParams = REFERENCE_DEFAULT, **kw) -> np.ndarray:
    """Fit und Anwendung in einem Schritt (pro Bild). Bei zu wenig Gewebe: Bild unverändert."""
    try:
        return macenko_apply(img, macenko_fit(img, **kw), ref)
    except (ValueError, np.linalg.LinAlgError):
        return img


def mosaic(patches: list[np.ndarray]) -> np.ndarray:
    """Hängt gleich grosse Patches untereinander – genügt für die Schätzung der Farbstatistik."""
    return np.concatenate([np.asarray(p)[..., :3] for p in patches], axis=0)


# --------------------------------------------------------------------------------------
# HED-Augmentation und kontrollierte Verschiebungen
# --------------------------------------------------------------------------------------

def _hed(img: np.ndarray) -> np.ndarray:
    return rgb2hed(img.astype(np.float64) / 255.0)


def _rgb(hed: np.ndarray) -> np.ndarray:
    return (np.clip(hed2rgb(hed), 0, 1) * 255).round().astype(np.uint8)


def hed_jitter(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Zufällige Färbevariante nach Tellez et al.: s' = alpha*s + beta je HED-Kanal.

    alpha ~ U(1-sigma, 1+sigma). beta ~ U(-sigma, sigma) * std(s) – der additive Teil wird
    relativ zur Streuung des Kanals skaliert, damit sigma unabhängig von der
    Skalierung des HED-Raums interpretierbar bleibt.
    """
    if sigma <= 0:
        return img
    hed = _hed(img)
    std = hed.reshape(-1, 3).std(axis=0) + 1e-8
    alpha = rng.uniform(1 - sigma, 1 + sigma, size=3)
    beta = rng.uniform(-sigma, sigma, size=3) * std
    return _rgb(hed * alpha + beta)


def hed_scale(img: np.ndarray, h_factor: float = 1.0, e_factor: float = 1.0) -> np.ndarray:
    """Deterministische Änderung der Färbeintensität: H- und E-Konzentration skalieren."""
    if h_factor == 1.0 and e_factor == 1.0:
        return img
    hed = _hed(img)
    hed[..., 0] *= h_factor
    hed[..., 1] *= e_factor
    return _rgb(hed)


class HEDShift:
    """Testzeit-Verschiebung mit festem Seed – alle Modelle sehen exakt dieselben Bilder.

    per='slide': alle Patches eines Slides bekommen dieselbe Verschiebung (wie ein anderes
    Labor oder ein anderer Färbelauf). per='image': jede Kachel eine eigene.
    """

    def __init__(self, sigma: float, seed: int = 0, per: str = "slide"):
        self.sigma, self.seed, self.per = sigma, seed, per

    def __call__(self, img: np.ndarray, idx: int, slide: int | None = None) -> np.ndarray:
        key = slide if (self.per == "slide" and slide is not None) else idx
        rng = np.random.default_rng(self.seed * 1_000_003 + int(key))
        alpha = rng.uniform(1 - self.sigma, 1 + self.sigma, size=3)
        u = rng.uniform(-self.sigma, self.sigma, size=3)
        if self.sigma <= 0:
            return img
        hed = _hed(img)
        std = hed.reshape(-1, 3).std(axis=0) + 1e-8
        return _rgb(hed * alpha + u * std)

    def __repr__(self) -> str:
        return f"HEDShift(sigma={self.sigma}, per={self.per})"


class HEDScaleShift:
    """Testzeit-Verschiebung: H und E um feste Faktoren skalieren (für alle Bilder gleich)."""

    def __init__(self, h_factor: float, e_factor: float):
        self.h, self.e = h_factor, e_factor

    def __call__(self, img: np.ndarray, idx: int = 0, slide: int | None = None) -> np.ndarray:
        return hed_scale(img, self.h, self.e)

    def __repr__(self) -> str:
        return f"HEDScaleShift(h={self.h}, e={self.e})"


# --------------------------------------------------------------------------------------
# Farbstatistik und Shift-Metriken
# --------------------------------------------------------------------------------------

def tissue_mask(img: np.ndarray, Io: float = 240, beta: float = 0.15) -> np.ndarray:
    """True für Gewebepixel: mittlere OD über die Kanäle >= beta.

    (Macenko selbst verwirft strenger jedes Pixel mit einem Kanal unter beta; für
    Farbstatistiken würde das eosinreiches Stroma unterrepräsentieren.)"""
    OD = _to_od(img, Io).reshape(img.shape)
    return OD.mean(axis=-1) >= beta


def lab_stats(img: np.ndarray) -> dict:
    """Mittelwert und Streuung von L*, a*, b* über Gewebepixel (Fallback: alle Pixel)."""
    lab = rgb2lab(img)
    m = tissue_mask(img)
    px = lab[m] if m.sum() >= 50 else lab.reshape(-1, 3)
    mu, sd = px.mean(axis=0), px.std(axis=0)
    return {"L_mean": float(mu[0]), "a_mean": float(mu[1]), "b_mean": float(mu[2]),
            "L_std": float(sd[0]), "a_std": float(sd[1]), "b_std": float(sd[2]),
            "tissue_frac": float(m.mean())}


def _median_gamma(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator, n: int = 500) -> float:
    Z = np.concatenate([X, Y])
    Z = Z[rng.choice(len(Z), size=min(n, len(Z)), replace=False)]
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    return 1.0 / max(med, 1e-12)


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float | None = None, seed: int = 0) -> float:
    """Maximum Mean Discrepancy (verzerrte Schätzung, RBF-Kern, Median-Heuristik).
    0 = gleiche Verteilung; grösser = stärker verschoben."""
    rng = np.random.default_rng(seed)
    if gamma is None:
        gamma = _median_gamma(X, Y, rng)

    def k(A, B):
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(d2, 0))

    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def frechet_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Fréchet-Distanz zweier Gauss-Approximationen (wie FID, aber auf beliebigen Merkmalen)."""
    mu1, mu2 = X.mean(0), Y.mean(0)
    s1, s2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if isinstance(covmean, tuple):  # ältere SciPy-Versionen
        covmean = covmean[0]
    covmean = np.real(covmean)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def pairwise_matrix(groups: dict, fn) -> tuple[list, np.ndarray]:
    """Symmetrische Distanzmatrix fn(X_a, X_b) für alle Gruppenpaare."""
    keys = sorted(groups)
    M = np.zeros((len(keys), len(keys)))
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            M[a, b] = M[b, a] = fn(groups[keys[a]], groups[keys[b]])
    return keys, M


def origin_predictability(features: np.ndarray, labels: np.ndarray, folds: int = 5,
                          seed: int = 0) -> dict:
    """Wie gut lässt sich die Herkunft (z. B. Spital) aus den Merkmalen vorhersagen?

    Balanced Accuracy einer logistischen Regression (Kreuzvalidierung).
    Zufallsniveau = 1 / Anzahl Gruppen. Deutlich darüber = starker Batch-Effekt.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, features, labels, cv=cv, scoring="balanced_accuracy")
    return {"balanced_accuracy": float(scores.mean()), "std": float(scores.std()),
            "chance": 1.0 / len(np.unique(labels))}
