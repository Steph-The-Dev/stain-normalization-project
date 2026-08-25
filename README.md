# Automated Stain Normalization in Digital Pathology

### Bridging Classical Imaging Physics and Modern Computer Vision

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Learning-red.svg)

## 📌 Overview

In digital pathology, the "Domain Shift" caused by variations in histological staining protocols (H&E) represents a significant hurdle for robust AI diagnostics. This project implements a high-precision normalization pipeline to ensure data consistency across different laboratory environments and sensor systems.

**Key Objective:** Minimize variance in histological slides by aligning color distributions in perceptually uniform color spaces.

![App Demo](assets/histological-stain-normalization.gif)

*(⬆️ Note: [See the live app for a full demonstration](https://stain-normalization-pro.streamlit.app/))*

---

## 🔬 Methodologies & Normalizer Strategies

This pipeline combines classical analytical color space transformations with modern deep generative networks via a polymorphic Strategy Pattern:

1. **Reinhard Normalization (CIELAB Color Space)**  
   Aligns global color distributions by matching channel-wise mean ($\mu$) and standard deviation ($\sigma$):
   $$P_{\text{out}} = (P_{\text{src}} - \mu_{\text{src}}) \cdot \frac{\sigma_{\text{trg}}}{\sigma_{\text{src}}} + \mu_{\text{trg}}$$
   *Includes morphological mask refinement and alpha blending to explicitly preserve the white glass background.*
2. **Macenko Normalization (Optical Density SVD Vector Decomposition)**  
   Decomposes RGB pixels into Optical Density (OD) space via SVD to extract hematoxylin & eosin stain vectors, matching maximum stain concentrations. *Integrates morphological mask refinement for cleaner stain vector extraction and background preservation blending.*
3. **Contrastive Unpaired Translation (CUT & SSIM Deep Learning)**  
   Uses a ResNet Generator with PatchNCE contrastive feature loss and structural SSIM loss. Includes Nearest-Neighbor upsampling to eliminate deconvolution checkerboards and continuous sigmoidal alpha matting ($15\times15$ Gaussian feathering) for seamless tissue-to-background transitions. *Features Cosine Annealing learning rate scheduling and Color Jitter augmentations for robust, fast convergence.*

---

## 🛠 Tech Stack

- **Python 3.9+**
- **OpenCV & NumPy:** For high-performance matrix operations and color space transformations.
- **PyTorch:** Transitioning the pipeline into a GPU-accelerated deep learning framework.
- **Matplotlib:** For histogram analysis and visual validation.

---

## ⚙️ Usage & Reproducibility

To ensure transparency and reproducibility, this repository separates the mathematical transformation logic from the presentation layer. It includes a dedicated visualization tool to generate standardized, publication-ready comparisons.

### 1. Run the Normalization Pipeline

Apply the Reinhard transformation to your raw data:

```bash
python normalization.py --source data/raw_slide.jpg --target data/gold_standard.jpg --output data/normalized_slide.jpg
```

### 2. Generate the Visual Comparison

Use the included `visualizer.py` script to create a high-resolution, 16:9 side-by-side comparison of the domain shift before and after the transformation.

```bash
python visualizer.py
```

_Note: Please ensure you do not commit raw clinical datasets to GitHub. Keep your image data in a local `data/` or `images/` directory that is added to your `.gitignore`._

---

## 🚀 Roadmap

- [x] Implementation of Reinhard Color Transfer (NumPy/OpenCV)
- [x] Implementation of Macenko SVD Stain Vector Matrix Normalizer
- [x] Quantitative Evaluation Benchmark (SSIM, PSNR, CIELAB Delta L/ab)
- [x] Automated batch processing for large-scale WSI (Whole Slide Imaging)
- [x] Integration of PyTorch-based Tensor processing & Unpaired Dataset Loader
- [x] Research: Unpaired Image-to-Image Translation using Contrastive Unpaired Translation (CUT) & SSIM Loss
- [x] Artifact-free Neural Generator Architecture (Nearest-Neighbor Upsampling, Continuous Sigmoidal Alpha Matting)
- [x] Algorithmic Quality Improvements (Tissue-Mask Background Preservation, Cosine Annealing LR, Color Jitter Augmentation)
- [x] ONNX Model Export & Streamlit Cloud Deployment Pipeline

---

## 👨‍💻 About the Author

**Stephan Pfeiffer** _Senior Imaging Expert | Data Science & Computer Vision Transition_ Combining 20 years of experience in high-end signal processing with cutting-edge AI methodologies.

- **Strategy:** "Bridge Builder" – Connecting Domain Expertise with AI Implementation.
- **Academic Path:** Preparing for **GSERM (University of St. Gallen)** and **MSc in Applied Information and Data Science (HSLU)**.
- **Portfolio:** [stephthedev.de](https://www.stephthedev.de)
