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
# TAB 2: BATCH PROCESSING
# ==========================================
with tab_batch:
    st.header("Batch Directory Rendering")
    st.info("Normalisiert alle Bilder in einem lokalen Ordner basierend auf einem Referenz-Look.")
    
    batch_target_file = st.file_uploader("Upload Target (Reference)", type=["jpg", "png", "tif"], key="trg_batch")
    
    col_paths1, col_paths2 = st.columns(2)
    with col_paths1:
        input_dir = st.text_input("Input Ordner-Pfad", value="data/raw/batch_input")
    with col_paths2:
        output_dir = st.text_input("Output Ordner-Pfad", value="data/processed/batch_output")
        
    col_bset1, col_bset2 = st.columns(2)
    with col_bset1:
        mask_method_batch = st.radio("Masking Method", ["HSV (Saturation)", "Luma (Grayscale)"], key="method_batch")
    with col_bset2:
        if mask_method_batch == "HSV (Saturation)":
            batch_threshold = st.slider("HSV Threshold", 0, 100, 15, key="slider_batch_hsv")
        else:
            batch_threshold = st.slider("Luma Threshold", 0, 255, 210, key="slider_batch_luma")

    if st.button("🚀 Start Batch Render") and batch_target_file:
        if not os.path.exists(input_dir):
            st.error(f"Ordner '{input_dir}' nicht gefunden!")
        else:
            target_img = load_uploaded_image(batch_target_file)
            os.makedirs(output_dir, exist_ok=True)
            
            valid_ext = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
            
            if not files:
                st.warning("Keine passenden Bilder im Input-Ordner gefunden.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(files):
                    status_text.text(f"Verarbeite: {file} ({i+1}/{len(files)})")
                    src_path = os.path.join(input_dir, file)
                    src_img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
                    
                    if src_img.dtype == 'uint16':
                        src_img = (src_img / 256).astype('uint8')
                        
                    # Methoden-Weiche für Batch
                    if mask_method_batch == "HSV (Saturation)":
                        res = normalize_stain_reinhard_hsv_final(src_img, target_img, src_sat_thresh=batch_threshold, target_sat_thresh=batch_threshold)
                    else:
                        res = normalize_stain_reinhard_custom(src_img, target_img, src_thresh=batch_threshold, target_thresh=batch_threshold)
                    
                    name, _ = os.path.splitext(file)
                    cv2.imwrite(os.path.join(output_dir, f"{name}_norm.tif"), res)
                    
                    progress_bar.progress((i + 1) / len(files))
                
                status_text.success(f"🎉 Batch Render abgeschlossen! {len(files)} Bilder gespeichert in {output_dir}.")


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