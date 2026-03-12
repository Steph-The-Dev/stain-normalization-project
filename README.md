# 🔬 Histological Stain Normalization Suite

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud_Ready-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green.svg)](https://opencv.org/)

**A cloud-ready, interactive computer vision tool bridging video post-production workflows and digital pathology. Designed to normalize H&E whole-slide images and WSI video feeds using smart tissue masking.**

![App Demo](assets/demo.gif) 
*(⬆️ Note: See the live app for a full demonstration)*

---

## 🧠 The Concept: Post-Production meets Pathology
In digital pathology, algorithms often fail because tissue scans (Whole Slide Images) from different laboratories exhibit massive color variations. While the statistical **Reinhard Method** is a standard approach to match these colors to a reference target, it often incorporates the bright white background (the glass slide) into its calculations, skewing the math.

**The Solution:** Drawing from video post-production (chroma keying), this tool generates a dynamic **HSV Saturation Mask** or **Luma Mask** to isolate the tissue *before* calculating the color statistics. 

## ✨ Key Features & Workflows

This suite offers three distinct workflows tailored for research and data engineering:

### 1. 📷 Single Image Look Dev
Interactive before-and-after preview. Upload a source and target image, switch between HSV/Luma keying, and dial in the perfect threshold via a real-time slider.

### 2. 📂 Cloud Batch Processing
Designed for bulk processing. Upload multiple images at once, dial in your mask settings using a live preview of the first image, and render the entire batch into a clean `.zip` archive.

### 3. 🎬 Smart Video Auto-Splicer (Two-Step Workflow)
For processing continuous WSI scanner feeds (e.g., Tissue Microarrays):
* **Step 1 (Analysis):** The engine reads the MP4 video, detects hard scene cuts using frame differencing, and physically splices the master file into individual sub-clips.
* **Step 2 (Individual Look Dev):** The UI generates a thumbnail gallery of all detected scenes. Users can apply individual HSV/Luma thresholds to *each specific scene* with a live preview.
* **Step 3 (Render):** Renders all color-corrected sub-clips and bundles them into a master `.zip` file.

---

## 🏗️ Architecture & Tech Stack
* **Core Logic:** Python, OpenCV, NumPy (Modularized in `src/reinhard.py`)
* **Frontend / GUI:** Streamlit (Cloud Deployable)
* **Quality Assurance:** `pytest` (Unit tests ensuring shape consistency and zero-division protection).

---

## 🚀 Run it yourself

### Option A: Try it live in the browser (Recommended)
The app is fully deployed on Streamlit Community Cloud. No installation required.
👉 **[INSERT_YOUR_STREAMLIT_CLOUD_LINK_HERE]**

### Option B: Local Installation
If you want to run the suite locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/stain-normalization-project.git](https://github.com/YOUR_USERNAME/stain-normalization-project.git)
   cd stain-normalization-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
---
*Developed as part of the MSc Applied Information and Data Science at HSLU.*