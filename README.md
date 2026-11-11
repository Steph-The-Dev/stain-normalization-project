# Automated Stain Normalization in Digital Pathology

### Bridging Classical Imaging Physics and Modern Computer Vision

[![tests](https://github.com/Steph-The-Dev/stain-normalization-project/actions/workflows/tests.yml/badge.svg)](https://github.com/Steph-The-Dev/stain-normalization-project/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)

**▶️ Live demo: [stain-normalization-pro.streamlit.app](https://stain-normalization-pro.streamlit.app/)**

![App Demo](assets/histological-stain-normalization.gif)

---

## 🩺 Why this matters

In digital pathology, histological images can differ substantially between
laboratories, scanners and staining protocols. These differences can act as
unintended signals for machine-learning models and reduce their robustness
when applied to data from a new domain.

This project demonstrates and compares stain-normalization approaches for
histopathological (H&E) images and provides an interactive tool for exploring
their effects.

**In short:**

| | |
|---|---|
| **Problem** | Color variation between labs/scanners ("domain shift") hurts the robustness of AI models in pathology. |
| **What this project does** | Implements three stain-normalization methods behind one common interface, plus quantitative metrics and an interactive Streamlit app to compare them. A separate experiment on Camelyon17-WILDS measures how much a new hospital costs a classifier and what normalization recovers ([`experiments/`](experiments/)). |
| **Methods compared** | Reinhard (CIELAB statistics), Macenko (optical-density stain vectors), CUT (contrastive unpaired deep learning). |
| **Status** | Research & demonstration tool — not clinically validated (see [Limitations](#️-limitations)). |

---

## 🚀 Quick start

```bash
git clone https://github.com/Steph-The-Dev/stain-normalization-project.git
cd stain-normalization-project

# Option A: conda
conda env create -f environment.local.yml
conda activate stain-norm-env

# Option B: pip / venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Run the test suite
pytest

# Launch the interactive app
streamlit run app.py
```

In the app, tick **"🧪 Use Built-in Demo Samples"** to try it immediately
with the bundled sample images in `data/raw/` — no upload required.

### Use it from Python

All normalizers share a scikit-learn-style `fit` / `transform` interface
(images are BGR `uint8` arrays as returned by OpenCV):

```python
import cv2
from src.normalizers import ReinhardNormalizer, MacenkoNormalizer, CUTStainNormalizer
from src.metrics import evaluate_normalization

source = cv2.imread("data/raw/source.tif")
target = cv2.imread("data/raw/target.tif")

normalizer = MacenkoNormalizer().fit(target)   # or ReinhardNormalizer(), CUTStainNormalizer()
result = normalizer.transform(source)

print(evaluate_normalization(source, result, target))
cv2.imwrite("normalized.png", result)
```

---

## 🔬 Methods

All three methods implement the same `BaseStainNormalizer` interface
(Strategy Pattern), so they are interchangeable in the app, in batch
processing and in evaluation.

1. **Reinhard Normalization (CIELAB Color Space)** — [`src/normalizers/reinhard.py`](src/normalizers/reinhard.py)  
   Aligns global color distributions by matching channel-wise mean ($\mu$) and standard deviation ($\sigma$):
   $$P_{\text{out}} = (P_{\text{src}} - \mu_{\text{src}}) \cdot \frac{\sigma_{\text{trg}}}{\sigma_{\text{src}}} + \mu_{\text{trg}}$$
   *Statistics are computed on tissue pixels only; morphological mask refinement and alpha blending preserve the white glass background.*
2. **Macenko Normalization (Optical Density SVD Decomposition)** — [`src/normalizers/macenko.py`](src/normalizers/macenko.py)  
   Converts RGB pixels into Optical Density (OD) space and uses SVD to estimate hematoxylin & eosin stain vectors, then matches stain concentrations to the target. *Uses the same tissue masking for cleaner stain-vector estimation and background preservation. Very dark pixels (brightest channel < 60) are excluded from the stain-vector estimate: in the Camelyon17 experiment they biased the hematoxylin vector in 16 of 50 slides.*
3. **Contrastive Unpaired Translation (CUT + SSIM)** — [`src/normalizers/cut.py`](src/normalizers/cut.py)  
   A ResNet generator trained with PatchNCE contrastive loss and a structural SSIM loss. Nearest-neighbor upsampling avoids checkerboard artifacts; Gaussian-feathered alpha matting gives smooth tissue-to-background transitions. Cosine-annealing LR and color-jitter augmentation for stable training. In the app, the model is trained on the fly on the selected image pair (CPU-friendly, small network).

**Supporting modules**

| Module | Purpose |
|---|---|
| [`src/tissue_mask.py`](src/tissue_mask.py) | HSV-saturation and luma-based tissue masks shared by all normalizers |
| [`src/metrics.py`](src/metrics.py) | SSIM, PSNR, CIELAB ΔL / Δab evaluation |
| [`src/patch_extractor.py`](src/patch_extractor.py) | Tissue-aware patch extraction from large slides |
| [`src/dataset.py`](src/dataset.py) | Unpaired PyTorch dataset for CUT training |
| [`src/export.py`](src/export.py) | ONNX export of the trained CUT generator |
| [`src/batch_process.py`](src/batch_process.py), [`src/video_process.py`](src/video_process.py) | Batch folder and video processing used by the app |

---

## 📊 Current results

**Interactive comparison (app).** The app compares the three methods visually
and statistically (SSIM, PSNR, CIELAB ΔL/Δab, RGB parade):

- Reinhard normalization reduces global color distribution differences, but also copies the tissue mix of the reference.
- Macenko normalization aligns the stain colors and keeps the stain amounts.
- CUT-based normalization can adapt local appearance, but may introduce
  artifacts or alter biologically relevant details.

**Cross-domain experiment ([`experiments/`](experiments/)).** Tumor/normal
classification on Camelyon17-WILDS patches with a ResNet-18, trained on
hospitals 0, 3, 4 and tested on hospitals 1 and 2. Accuracy in %, mean of
three runs:

| Training | Variant | Known hospitals | Unseen hospitals |
|---|---|---|---|
| From scratch | No protection | 95.0 | 74.1 |
| | HED color augmentation | 92.2 | 82.4 |
| | Macenko | 95.1 | 86.6 |
| ImageNet-pretrained | No protection | 85.7 | 88.7 |
| | HED color augmentation | 93.2 | 90.8 |
| | Macenko | 91.9 | 91.4 |

- Without pretraining, a new hospital costs 21 points; Macenko recovers 12.5, almost all of it in the color outlier (hospital 2).
- With pretraining the gap disappears. Macenko still adds 2.7 points on average, but slightly lowers accuracy in hospital 1.
- Under artificial stain shifts (σ = 0.5), 17.1 % of predictions flip without protection, 7.6 % with HED augmentation and 1.9 % with Macenko.

This is a small replication on one public dataset, not a clinical study.
Details, limitations and all numbers: [`experiments/README.md`](experiments/README.md) (in German).

---

## ⚠️ Limitations

This project is a research and demonstration tool, not a diagnostic system.
Stain normalization does not automatically improve diagnostic performance:
in the cross-domain experiment it helps on average but lowers accuracy slightly
in one of the two unseen hospitals.
It can also reduce biologically meaningful color variation if applied too
aggressively.

The current implementation has not been clinically validated and should not
be used for patient care or diagnostic decision-making.

Further technical limitations:

- Results depend on the choice of the target (reference) image.
- The CUT model in the app is trained on a single image pair for a few epochs; it is a demonstration, not a production-grade model.
- Tissue masking uses simple thresholds and may need tuning for unusual stains or scanners.

---

## 🧪 Testing

```bash
pytest
```

The test suite covers the normalizers, metrics, tissue masks, dataset loader,
patch extraction, the CUT model and ONNX export. Tests run automatically on
every push via [GitHub Actions](.github/workflows/tests.yml).

> **Data note:** Please do not commit clinical datasets. Only the two small
> demo images in `data/raw/` are tracked; everything else in `data/` is
> git-ignored.

---

## 🛠 Tech Stack

- **Python 3.10+**
- **OpenCV & NumPy** — color space transformations and matrix operations
- **PyTorch** — CUT generator, PatchNCE and SSIM losses; ONNX export
- **Streamlit** — interactive comparison app
- **pytest & GitHub Actions** — automated testing

---

## 🗺 Roadmap

- [x] Implementation of Reinhard Color Transfer (NumPy/OpenCV)
- [x] Implementation of Macenko SVD Stain Vector Matrix Normalizer
- [x] Quantitative Evaluation Benchmark (SSIM, PSNR, CIELAB Delta L/ab)
- [x] Batch processing of image folders and tissue-aware patch extraction
- [x] Integration of PyTorch-based Tensor processing & Unpaired Dataset Loader
- [x] Research: Unpaired Image-to-Image Translation using Contrastive Unpaired Translation (CUT) & SSIM Loss
- [x] Generator without checkerboard artifacts (Nearest-Neighbor Upsampling, Continuous Sigmoidal Alpha Matting)
- [x] Algorithmic Quality Improvements (Tissue-Mask Background Preservation, Cosine Annealing LR, Color Jitter Augmentation)
- [x] ONNX Model Export & Streamlit Cloud Deployment Pipeline
- [x] Cross-domain evaluation on Camelyon17-WILDS: Macenko vs. HED augmentation on a downstream classifier ([`experiments/`](experiments/))
- [x] Dark-pixel filter for robust Macenko stain-vector estimation
- [ ] Reinhard and CUT in the cross-domain evaluation

---

## 👨‍💻 About the Author

**Stephan Pfeiffer** | _Senior Imaging & Color Science Expert | MSc Candidate in Applied Data Science & AI (HSLU)_

More than 20 years as a colorist and editor in broadcast and film post-production: matching cameras, color spaces and signal chains. This project applies that experience to color variation in digital pathology.

- **Studies:** MSc in Applied Data Science and AI at HSLU (since September 2026); "Deep Learning: Fundamentals & Applications" at GSERM (University of St. Gallen).
- **Background:** Dipl.-Kaufmann (International Business Studies). English and Spanish.
- **Portfolio:** [stephthedev.de](https://www.stephthedev.de)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
