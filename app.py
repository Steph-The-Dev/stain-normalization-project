import streamlit as st
import cv2
import numpy as np
import os
import tempfile

# Importiere BEIDE Kern-Funktionen
from src.reinhard import normalize_stain_reinhard_hsv_final, normalize_stain_reinhard_custom

# --- CONFIG ---
st.set_page_config(page_title="Stain Normalization Pro", page_icon="🔬", layout="wide")

# Hilfsfunktion für den Bild-Upload
def load_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

# --- HEADER ---
st.title("🔬 Histological Stain Normalization Suite")
st.markdown("**Powered by Reinhard Method & Smart Tissue Masking (HSV/Luma)**")

# --- UI TABS ---
tab_single, tab_batch, tab_video = st.tabs(["📷 Single Image", "📂 Batch Processing", "🎬 Video Analysis"])

# ==========================================
# TAB 1: SINGLE IMAGE
# ==========================================
with tab_single:
    st.header("Single Image Grading")
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        source_file = st.file_uploader("Upload Source Image", type=["jpg", "png", "tif"], key="src_single")
    with col_upload2:
        target_file = st.file_uploader("Upload Target (Reference)", type=["jpg", "png", "tif"], key="trg_single")

    st.markdown("### Keying Settings")
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        mask_method_single = st.radio("Masking Method", ["HSV (Saturation)", "Luma (Grayscale)"], key="method_single")
    with col_set2:
        if mask_method_single == "HSV (Saturation)":
            threshold_single = st.slider("HSV Threshold (Tissue > X)", 0, 100, 15, key="slider_single_hsv")
        else:
            threshold_single = st.slider("Luma Threshold (Tissue < X)", 0, 255, 210, key="slider_single_luma")

    if source_file and target_file:
        src_img = load_uploaded_image(source_file)
        target_img = load_uploaded_image(target_file)

        with st.spinner("Rendering..."):
            if mask_method_single == "HSV (Saturation)":
                result_img = normalize_stain_reinhard_hsv_final(src_img, target_img, src_sat_thresh=threshold_single, target_sat_thresh=threshold_single)
            else:
                result_img = normalize_stain_reinhard_custom(src_img, target_img, src_thresh=threshold_single, target_thresh=threshold_single)

        c1, c2, c3 = st.columns(3)
        c1.image(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB), caption="1. Source", width='stretch')
        c2.image(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB), caption="2. Target", width='stretch')
        c3.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="3. Result", width='stretch')


# ==========================================
# TAB 2: BATCH PROCESSING (Cloud Ready via ZIP)
# ==========================================
import zipfile
import io

with tab_batch:
    st.header("Cloud Batch Rendering")
    st.info("Lade mehrere Bilder gleichzeitig hoch. Der Server verarbeitet sie und schnürt dir ein ZIP-Paket zum Download.")
    
    col_batch_up1, col_batch_up2 = st.columns(2)
    with col_batch_up1:
        # NEU: accept_multiple_files=True erlaubt das Markieren von hunderten Bildern!
        source_files = st.file_uploader("Upload Source Images (Mehrere markieren)", type=["jpg", "png", "tif"], accept_multiple_files=True, key="src_batch")
    with col_batch_up2:
        batch_target_file = st.file_uploader("Upload Target (Reference)", type=["jpg", "png", "tif"], key="trg_batch")
        
    col_bset1, col_bset2 = st.columns(2)
    with col_bset1:
        mask_method_batch = st.radio("Masking Method", ["HSV (Saturation)", "Luma (Grayscale)"], key="method_batch")
    with col_bset2:
        if mask_method_batch == "HSV (Saturation)":
            batch_threshold = st.slider("HSV Threshold", 0, 100, 15, key="slider_batch_hsv")
        else:
            batch_threshold = st.slider("Luma Threshold", 0, 255, 210, key="slider_batch_luma")

    if st.button("🚀 Start Batch Render") and batch_target_file and source_files:
        target_img = load_uploaded_image(batch_target_file)
        
        # Ein virtuelles ZIP-Archiv im Arbeitsspeicher des Servers erstellen
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(source_files):
                status_text.text(f"Verarbeite: {file.name} ({i+1}/{len(source_files)})")
                
                # Bild aus dem RAM laden
                src_img = load_uploaded_image(file)
                
                # Grading anwenden
                if mask_method_batch == "HSV (Saturation)":
                    res = normalize_stain_reinhard_hsv_final(src_img, target_img, src_sat_thresh=batch_threshold, target_sat_thresh=batch_threshold)
                else:
                    res = normalize_stain_reinhard_custom(src_img, target_img, src_thresh=batch_threshold, target_thresh=batch_threshold)
                
                # Fertiges Bild wieder in Bytes umwandeln (als hochwertiges PNG)
                is_success, buffer = cv2.imencode(".png", res)
                if is_success:
                    # Bild in das ZIP-Archiv schreiben
                    original_name, _ = os.path.splitext(file.name)
                    zip_file.writestr(f"{original_name}_normalized.png", buffer.tobytes())
                
                progress_bar.progress((i + 1) / len(source_files))
            
            status_text.success("🎉 Batch Render abgeschlossen! Lade dein ZIP-Archiv herunter.")
            
        # Wenn die Schleife fertig ist, den Download-Button anzeigen
        st.download_button(
            label="💾 Download Normalized Images (.zip)",
            data=zip_buffer.getvalue(),
            file_name="normalized_batch.zip",
            mime="application/zip"
        )

# ==========================================
# TAB 3: VIDEO ANALYSIS & SCENE DETECTION
# ==========================================
with tab_video:
    st.header("Video Processing & Cut Detection")
    
    col_vid1, col_vid2 = st.columns(2)
    with col_vid1:
        video_file = st.file_uploader("Upload WSI Video (MP4)", type=["mp4", "avi", "mov"])
    with col_vid2:
        vid_target_file = st.file_uploader("Upload Target (Reference)", type=["jpg", "png", "tif"], key="trg_vid")

    st.markdown("### Settings")
    col_vset1, col_vset2, col_vset3 = st.columns(3)
    with col_vset1:
        mask_method_vid = st.radio("Masking Method", ["HSV (Saturation)", "Luma (Grayscale)"], key="method_vid")
    with col_vset2:
        if mask_method_vid == "HSV (Saturation)":
            vid_mask_thresh = st.slider("HSV Threshold", 0, 100, 15, key="slider_vid_hsv")
        else:
            vid_mask_thresh = st.slider("Luma Threshold", 0, 255, 210, key="slider_vid_luma")
    with col_vset3:
        scene_thresh = st.slider("Scene Cut Sensitivity", 10.0, 100.0, 43.5, step=0.5)

    if video_file and vid_target_file:
        if st.button("🎬 Start Video Engine"):
            target_img = load_uploaded_image(vid_target_file)
            
            tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile_in.write(video_file.read())
            tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            out_path = tfile_out.name
            
            cap = cv2.VideoCapture(tfile_in.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            prev_gray = None
            cuts = []
            cooldown = 0
            
            vid_progress = st.progress(0)
            vid_status = st.empty()
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Cut Detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if cooldown > 0:
                    cooldown -= 1
                elif prev_gray is not None:
                    diff = np.mean(cv2.absdiff(gray, prev_gray))
                    if diff > scene_thresh:
                        cuts.append((frame_idx, frame_idx/fps, diff))
                        cooldown = 15
                prev_gray = gray
                
                # Normalization
                try:
                    if mask_method_vid == "HSV (Saturation)":
                        norm_frame = normalize_stain_reinhard_hsv_final(frame, target_img, src_sat_thresh=vid_mask_thresh, target_sat_thresh=vid_mask_thresh)
                    else:
                        norm_frame = normalize_stain_reinhard_custom(frame, target_img, src_thresh=vid_mask_thresh, target_thresh=vid_mask_thresh)
                except:
                    norm_frame = frame
                
                out.write(norm_frame)
                
                frame_idx += 1
                if frame_idx % 10 == 0 or frame_idx == total_frames:
                    vid_progress.progress(frame_idx / total_frames)
                    vid_status.text(f"Rendere Frame {frame_idx} / {total_frames}...")
                    
            cap.release()
            out.release()
            
            vid_status.success("Render Complete!")
            
            with open(out_path, 'rb') as f:
                st.download_button("💾 Download Normalized Video", f, file_name="normalized_video.mp4", mime="video/mp4")
            
            st.subheader("📊 Detected Scene Cuts")
            st.metric("Total Cuts", len(cuts))
            if cuts:
                for c in cuts:
                    st.write(f"✂️ **Cut bei {c[1]:.2f}s** (Frame {c[0]}) | Diff: {c[2]:.1f}")