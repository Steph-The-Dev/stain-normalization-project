# 🔬 Histological Stain Normalization Suite
**Post-Production Workflows for Digital Pathology**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud_Ready-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green.svg)](https://opencv.org/)

A cloud-ready, highly interactive computer vision tool bridging the gap between video post-production pipelines and medical data engineering. Designed to normalize H&E whole-slide images (WSI) and scanner video feeds using smart tissue masking and real-time proxy rendering.

![App Demo](assets/demo.gif) 
*(⬆️ Note: See the live app for a full demonstration)*

---

## 🧠 The Concept: Why Post-Production?
In digital pathology, algorithms often fail because tissue scans from different laboratories exhibit massive color variations. While the statistical **Reinhard Method** is a standard approach for color matching, it has two major flaws:
1. It incorporates the bright white background (glass slide) into its statistics.
2. It completely overwrites the natural luminance variance (micro-contrast) of the source tissue, often leading to clipped highlights and crushed shadows.

**The Solution:** This suite introduces professional compositing and color grading techniques to data science:
* **Chroma/Luma Keying:** Generates dynamic HSV or Grayscale masks to isolate tissue *before* calculating color statistics.
* **Luma Preservation (Blend Node):** Acts as an opacity slider for the L-channel (LAB color space), allowing users to perfectly mix the target's color (Hue/Sat) with the source's natural micro-contrast.

---

## ✨ Key Features & Workflows

### 1. 📷 Single Image Look Dev (Real-Time)
* **Proxy-Workflow:** UI renders on scaled-down proxy images, reducing matrix calculation latency from ~500ms to <10ms.
* **Hardware-Style Scopes:** Custom built, ultra-fast 2D-Histogram RGB Parades (rendered via pure NumPy/OpenCV) update in real-time alongside slider inputs.
* **High-Res Export:** Dedicated render engine applies look-dev settings to the original 4K master files for a lossless PNG download.

### 2. 📂 Cloud Batch Processing
Upload hundreds of images at once. Dial in your keying and luma settings using the live proxy preview of the first image, then deploy the render job across the entire batch. Exports as a clean `.zip` archive.

### 3. 🎬 Smart Video Auto-Splicer (Two-Step Workflow)
Designed for continuous WSI scanner feeds (e.g., Tissue Microarrays):
* **Step 1 (Analysis):** Detects hard scene cuts via frame differencing and splices the master video into individual sub-clips.
* **Step 2 (Individual Look Dev):** Generates a thumbnail UI for every detected scene. Users can apply individual Keying and Luma Preservation settings *per clip*.
* **Step 3 (Render):** Renders all normalized sub-clips at full resolution and bundles them into a master `.zip`.

---

## 🏗️ Architecture & Tech Stack
* **Core Logic:** `Python`, `OpenCV`, `NumPy` (Modularized in `src/reinhard.py`)
* **Frontend:** `Streamlit` utilizing `@st.fragment` caching for localized, zero-latency DOM updates.
* **Quality Assurance:** `pytest` (Unit tests ensuring shape consistency, zero-division protection, and dual-engine coverage).

---

## 🚀 Run it yourself

### Try it live in the browser (Recommended)
Fully deployed on Streamlit Community Cloud. No installation required.
👉 **[[Stain Normalization App auf Streamlit](https://stain-normalization-pro.streamlit.app/)]**

### Local Installation
```bash
git clone [https://github.com/Steph-The-Dev/stain-normalization-project.git](https://github.com/Steph-The-Dev/stain-normalization-project.git)
cd stain-normalization-project
pip install -r requirements.txt
streamlit run app.py
```

---
*Developed as part of the MSc Applied Information and Data Science at HSLU.*