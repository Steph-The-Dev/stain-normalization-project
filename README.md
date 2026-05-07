# Automated Stain Normalization in Digital Pathology

### Bridging Classical Imaging Physics and Modern Computer Vision

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Learning-red.svg)

## 📌 Overview

In digital pathology, the "Domain Shift" caused by variations in histological staining protocols (H&E) represents a significant hurdle for robust AI diagnostics. This project implements a high-precision normalization pipeline to ensure data consistency across different laboratory environments and sensor systems.

**Key Objective:** Minimize variance in histological slides by aligning color distributions in perceptually uniform color spaces.

---

## 🔬 The "Bridge Builder" Approach

Unlike "black-box" approaches, this pipeline leverages **Imaging Physics** and **Statistical Signal Processing**:

1.  **Classical Foundation:** Implementation of the **Reinhard Method** for global color transfer.
2.  **Color Science:** Utilizing the **CIELAB color space** to decouple luminance from chromaticity—minimizing structural artifacts during normalization.
3.  **Future Evolution:** Scaling the pipeline towards **Generative Adversarial Networks (CycleGANs)** to handle non-linear stain variations.

---

## 🧮 Methodology: The Reinhard Transformation

The core of this implementation is a statistical alignment of the source image ($S$) to a target "Gold Standard" ($T$). We perform a linear transformation of the color distribution for each channel:

$$P_{out} = (P_{src} - \mu_{src}) \cdot \frac{\sigma_{trg}}{\sigma_{src}} + \mu_{trg}$$

- **Centering:** Removing the source mean ($\mu_{src}$).
- **Rescaling:** Matching the target standard deviation ($\sigma_{trg}$).
- **Shifting:** Aligning to the target mean ($\mu_{trg}$).

---

## 🛠 Tech Stack

- **Python 3.9+**
- **OpenCV & NumPy:** For high-performance matrix operations and color space transformations.
- **PyTorch:** Transitioning the pipeline into a GPU-accelerated deep learning framework.
- **Matplotlib:** For histogram analysis and visual validation.

---

## 📊 Results

|                    Original Slide (Source)                    |                        Target Template                        |                     Normalized Result                      |
| :-----------------------------------------------------------: | :-----------------------------------------------------------: | :--------------------------------------------------------: |
| ![Source](https://via.placeholder.com/200?text=Stained+Slide) | ![Target](https://via.placeholder.com/200?text=Gold+Standard) | ![Result](https://via.placeholder.com/200?text=Normalized) |

_(Note: High-resolution comparisons will be added as the project progresses.)_

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
- [x] Automated batch processing for large-scale WSI (Whole Slide Imaging)
- [ ] Integration of PyTorch-based Tensor processing
- [ ] Research: Unpaired Image-to-Image Translation using CycleGANs

---

## 👨‍💻 About the Author

**Stephan Pfeiffer** _Senior Imaging Expert | Data Science & Computer Vision Transition_ Combining 20 years of experience in high-end signal processing with cutting-edge AI methodologies.

- **Strategy:** "Bridge Builder" – Connecting Domain Expertise with AI Implementation.
- **Academic Path:** Preparing for **GSERM (University of St. Gallen)** and **MSc in Applied Information and Data Science (HSLU)**.
- **Portfolio:** [stephthedev.de](https://www.stephthedev.de)
