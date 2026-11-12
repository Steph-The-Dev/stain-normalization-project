"""
Fetch high-resolution demo tiles for the Streamlit app from CAMELYON17 (CC0).

Reads only the tiles it needs from the whole-slide images via HTTP range
requests, so a 1024 x 1024 crop costs a few MB instead of a 2-3 GB download.

Default pair = a real domain shift from the same dataset as the experiments:
    source: patient_040_node_0  (centre 2, UMCU - the WILDS test centre)
    target: patient_000_node_0  (centre 0, RUMC - a WILDS training centre)

Usage
-----
    pip install tifffile imagecodecs "fsspec[http]" aiohttp
    python scripts/fetch_demo_tiles.py                     # writes data/raw/source.tif + target.tif
    python scripts/fetch_demo_tiles.py --size 1536 --level 0
    python scripts/fetch_demo_tiles.py --source patient_041_node_2 --target patient_062_node_3

Check slide names with:
    aws s3 ls --no-sign-request s3://camelyon-dataset/CAMELYON17/images/

Data: CAMELYON dataset, Litjens et al., GigaScience 2018, doi:10.1093/gigascience/giy065.
Accessed via https://registry.opendata.aws/camelyon (licence: CC0).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import fsspec
import numpy as np
import tifffile

DEFAULT_BASE_URL = "https://camelyon-dataset.s3.us-west-2.amazonaws.com/CAMELYON17/images"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiled_levels(tif: tifffile.TiffFile) -> list[tifffile.TiffPage]:
    """Return the tiled RGB pages of the pyramid, largest first."""
    pages = [p for p in tif.pages if p.is_tiled and p.samplesperpixel >= 3]
    return sorted(pages, key=lambda p: p.imagewidth, reverse=True)


def read_region(page: tifffile.TiffPage, fh, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Read an RGB region from a tiled TIFF page, fetching only the needed tiles."""
    tw, th = page.tilewidth, page.tilelength
    tiles_x = math.ceil(page.imagewidth / tw)
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    decode = page.decode
    jpegtables = getattr(page, "jpegtables", None)

    for ty in range(y // th, (y + h - 1) // th + 1):
        for tx in range(x // tw, (x + w - 1) // tw + 1):
            idx = ty * tiles_x + tx
            nbytes = page.databytecounts[idx]
            if nbytes == 0:          # sparse / empty tile -> stays white
                continue
            fh.seek(page.dataoffsets[idx])
            data = fh.read(nbytes)
            tile, _, _ = decode(data, idx, jpegtables=jpegtables)
            tile = np.asarray(tile).reshape(th, tw, -1)[..., :3]

            # Intersection of this tile with the requested region
            x0, y0 = tx * tw, ty * th
            ix0, iy0 = max(x, x0), max(y, y0)
            ix1, iy1 = min(x + w, x0 + tw), min(y + h, y0 + th)
            out[iy0 - y:iy1 - y, ix0 - x:ix1 - x] = tile[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    return out


def pick_tissue_location(thumb_rgb: np.ndarray, frac: float, seed: int) -> tuple[float, float]:
    """
    Pick the centre (relative coords 0..1) of a dense tissue region on the thumbnail.
    'frac' = tile size relative to thumbnail width. Prefers the densest windows,
    picks randomly among the top 5 % so different seeds give different crops.
    """
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    tissue = ((hsv[..., 1] > 20) & (hsv[..., 2] < 235)).astype(np.float32)
    k = max(3, int(frac * thumb_rgb.shape[1]) | 1)
    density = cv2.blur(tissue, (k, k))
    # keep away from the border so the crop fits
    m = k // 2 + 1
    density[:m, :] = density[-m:, :] = 0
    density[:, :m] = density[:, -m:] = 0
    if density.max() < 0.5:
        raise RuntimeError("No sufficiently dense tissue found on this slide - try another one.")
    thresh = np.quantile(density[density > 0], 0.95)
    ys, xs = np.nonzero(density >= max(thresh, 0.85 * density.max()))
    i = np.random.default_rng(seed).integers(len(ys))
    return xs[i] / thumb_rgb.shape[1], ys[i] / thumb_rgb.shape[0]


def fetch_tile(url: str, size: int, level: int, seed: int) -> np.ndarray:
    with fsspec.open(url, "rb", block_size=2**20) as fh, tifffile.TiffFile(fh) as tif:
        levels = _tiled_levels(tif)
        if level >= len(levels):
            raise ValueError(f"Slide has only {len(levels)} tiled levels.")
        page = levels[level]

        # Smallest level that is still >= 1000 px wide serves as thumbnail
        thumb_page = next((p for p in reversed(levels) if p.imagewidth >= 1000), levels[-1])
        thumb = read_region(thumb_page, fh, 0, 0, thumb_page.imagewidth, thumb_page.imagelength)

        cx, cy = pick_tissue_location(thumb, size / page.imagewidth, seed)
        x = int(np.clip(cx * page.imagewidth - size / 2, 0, page.imagewidth - size))
        y = int(np.clip(cy * page.imagelength - size / 2, 0, page.imagelength - size))
        print(f"  level {level}: {page.imagewidth}x{page.imagelength}, crop at ({x}, {y})")
        return read_region(page, fh, x, y, size, size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="patient_040_node_0")
    ap.add_argument("--target", default="patient_000_node_0")
    ap.add_argument("--size", type=int, default=1024, help="edge length in px (default 1024)")
    ap.add_argument("--level", type=int, default=1,
                    help="pyramid level: 0 = ~0.24 um/px (40x), 1 = ~0.48 um/px (20x, default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for role, slide in (("source", args.source), ("target", args.target)):
        print(f"{role}: {slide}")
        rgb = fetch_tile(f"{args.base_url}/{slide}.tif", args.size, args.level, args.seed)
        out = args.out_dir / f"{role}.tif"
        cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))  # OpenCV writes LZW-compressed TIFF
        print(f"  -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
