import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Importiere DEIN EIGENES Modul!
from src.reinhard import normalize_stain_reinhard_custom, normalize_stain_reinhard_hsv_final

# --- Seiten-Konfiguration ---
st.set_page_config(page_title="Stain Normalization Pro", layout="wide")

st.title("🔬 Interaktive H&E Stain Normalization")
st.write("Lade deine Schnittbilder hoch und passe die Gewebe-Maske live an.")

# --- SEITENLEISTE (Werkzeuge & Upload) ---
st.sidebar.header("⚙️ Inspector / Controls")

# 1. Bild-Upload
st.sidebar.subheader("1. Media Pool")
source_file = st.sidebar.file_uploader("Source Image (Original)", type=["jpg", "jpeg", "png", "tif", "tiff"])
target_file = st.sidebar.file_uploader("Target Image (Referenz-Look)", type=["jpg", "jpeg", "png", "tif", "tiff"])

st.sidebar.divider()

# 2. Keyer-Einstellungen (Luma vs. HSV)
st.sidebar.subheader("2. Keying (Maske)")
mask_method = st.sidebar.radio("Maskierungs-Methode:", ("HSV (Sättigung)", "Luma (Graustufen)"))

# Dynamischer Slider, je nachdem was ausgewählt wurde
if mask_method == "HSV (Sättigung)":
    threshold = st.sidebar.slider("Sättigungs-Schwellenwert", min_value=0, max_value=255, value=15)
    st.sidebar.caption("Empfohlen für H&E. Höherer Wert = Striktere Gewebe-Erkennung.")
else:
    threshold = st.sidebar.slider("Helligkeits-Schwellenwert", min_value=0, max_value=255, value=210)
    st.sidebar.caption("Niedriger Wert = Striktere Erkennung (ignoriert helles Gewebe).")


# --- HILFSFUNKTION FÜR DEN UPLOAD ---
def load_uploaded_image(uploaded_file):
    """Konvertiert den Streamlit-Upload in ein OpenCV BGR-Bild."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)


# --- HAUPTBEREICH (Der Viewer) ---
if source_file and target_file:
    # Bilder in den RAM laden
    src_img = load_uploaded_image(source_file)
    target_img = load_uploaded_image(target_file)

    # Lade-Spinner anzeigen, während gerechnet wird
    with st.spinner('Rendere Frame...'):
        # Die entsprechende Funktion aus deinem Modul aufrufen
        if mask_method == "HSV (Sättigung)":
            result_img = normalize_stain_reinhard_hsv_final(
                src_img, target_img, src_sat_thresh=threshold, target_sat_thresh=threshold
            )
        else:
            result_img = normalize_stain_reinhard_custom(
                src_img, target_img, src_thresh=threshold, target_thresh=threshold
            )

    # OpenCV (BGR) zu RGB für die Web-Darstellung umwandeln
    src_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Layout: 3 Spalten nebeneinander
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Source (Eingabe)")
        st.image(src_rgb, width='stretch')

    with col2:
        st.subheader("Target (Referenz)")
        st.image(target_rgb, width='stretch')

    with col3:
        st.subheader("Result (Normalisiert)")
        st.image(result_rgb, width='stretch')
        
    st.success("Rendering erfolgreich! Ändere den Regler links, um das Ergebnis live anzupassen.")

else:
    # Startbildschirm, wenn noch keine Bilder hochgeladen wurden
    st.info("👈 Bitte lade in der Seitenleiste ein Source- und ein Target-Bild hoch.")