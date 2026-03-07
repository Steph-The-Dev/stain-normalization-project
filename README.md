# 🔬 Histological Stain Normalization (Reinhard Method)

**A modular Python tool with an interactive Streamlit GUI for automated color normalization of H&E stained whole-slide images.**

![App Demo](assets/demo.gif) 
*(⬆️ Ersetze diesen Platzhalter später mit einem echten GIF deiner App, das du im Ordner `assets/` speicherst)*

Dieses Projekt entstand im Rahmen der Vorbereitung auf den MSc Applied Information and Data Science an der HSLU. Ziel ist es, Farbvarianzen in der digitalen Pathologie (verursacht durch unterschiedliche Scanner und Färbeprotokolle) zu standardisieren, um eine robuste Datengrundlage für Machine-Learning-Modelle zu schaffen.

## 🧠 Methodik & Domain-Spezifik
Das Tool nutzt die statistische **Reinhard-Methode** im LAB-Farbraum. 
Die Besonderheit dieser Implementierung liegt im dynamischen **Tissue-Masking**:
Um zu verhindern, dass das weiße Trägerglas (Hintergrund) die globalen Bildstatistiken verfälscht, generiert der Algorithmus automatisch eine Maske (wahlweise Luma-Key oder HSV Chroma-Key). Die Helligkeits- und Farbkorrektur wird somit exklusiv auf Basis der tatsächlichen Gewebe-Pixel berechnet.

## 🛠 Features
* **Interaktive Web-GUI:** Live-Vorschau und Parameter-Tuning mittels Streamlit.
* **Modularer Aufbau:** Kern-Logik sauber getrennt in `src/reinhard.py`.
* **Smart Masking:** Luma-Key (Graustufen) und HSV-Key (Sättigung) auswählbar.
* **Multi-Format Support:** Akzeptiert `.jpg`, `.png`, `.tif` etc. automatisch.
* **Automated QC:** Unit-Tests (via `pytest`) garantieren mathematische Stabilität (Zero-Division Protection, Shape/Type-Consistency).

## 💻 Tech-Stack
* **Sprache:** Python 3.10
* **Framework:** Streamlit (Web-GUI)
* **Bibliotheken:** OpenCV, NumPy, Matplotlib, Pytest
* **Umgebung:** Anaconda / Jupyter Notebooks

## 🚀 Setup & Ausführung

**1. Installation**
```bash
git clone https://github.com/Steph-The-Dev/stain-normalization-project.git
conda env create -f environment.yml
conda activate stain_norm_env
```

**2. App starten (GUI Mode)**
```bash
streamlit run app.py
```

**3. Tests ausführen (QC Mode)**
```bash
python -m pytest
```

🗺 Roadmap

    [x] Basis-Implementierung Reinhard-Methode

    [x] Tissue-Masking (Luma & HSV)

    [x] Modulare Architektur & Unit Tests

    [x] Streamlit Web-GUI

    [ ] Next: Video-Skalierung (Verarbeitung von Master-Files)

    [ ] Next: Automated Scene Detection für Schnitt-Erkennung in WSI-Videos
