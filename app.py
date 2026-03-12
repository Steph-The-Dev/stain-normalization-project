import streamlit as st
import cv2
import numpy as np
import os
import tempfile

# Importiere BEIDE Kern-Funktionen
from src.reinhard import normalize_stain_reinhard_hsv_final, normalize_stain_reinhard_custom

# --- CONFIG ---
st.set_page_config(page_title="Stain Normalization Pro", page_icon="🔬", layout="wide")

# --- SESSION STATE SETUP (Für den 2-Stufen Video Workflow) ---
if 'vid_scenes' not in st.session_state:
    st.session_state.vid_scenes = []
if 'vid_step' not in st.session_state:
    st.session_state.vid_step = 1

def reset_vid_state():
    """Setzt den Video-Speicher zurück, wenn ein neues Video hochgeladen wird."""
    st.session_state.vid_scenes = []
    st.session_state.vid_step = 1

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
# TAB 2: BATCH PROCESSING (Mit Preview & ZIP)
# ==========================================
import zipfile
import io

with tab_batch:
    st.header("Cloud Batch Rendering")
    st.info("Lade mehrere Bilder hoch. Stelle den Regler anhand der Live-Vorschau des ersten Bildes ein.")
    
    col_batch_up1, col_batch_up2 = st.columns(2)
    with col_batch_up1:
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

    # --- NEU: DIE LIVE-VORSCHAU ---
    if source_files and batch_target_file:
        st.markdown("### 👁️ Look Dev: Live-Vorschau (Erstes Bild)")
        target_img = load_uploaded_image(batch_target_file)
        preview_src = load_uploaded_image(source_files[0])
        
        # Grading für die Vorschau
        if mask_method_batch == "HSV (Saturation)":
            preview_res = normalize_stain_reinhard_hsv_final(preview_src, target_img, src_sat_thresh=batch_threshold, target_sat_thresh=batch_threshold)
        else:
            preview_res = normalize_stain_reinhard_custom(preview_src, target_img, src_thresh=batch_threshold, target_thresh=batch_threshold)
            
        c1, c2, c3 = st.columns(3)
        c1.image(cv2.cvtColor(preview_src, cv2.COLOR_BGR2RGB), caption=f"Source: {source_files[0].name}", width='stretch')
        c2.image(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB), caption="Target", width='stretch')
        c3.image(cv2.cvtColor(preview_res, cv2.COLOR_BGR2RGB), caption="Result Preview", width='stretch')
        
        st.divider()

        # --- DER RENDER PROZESS ---
        if st.button("🚀 Start Full Batch Render"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(source_files):
                    status_text.text(f"Verarbeite: {file.name} ({i+1}/{len(source_files)})")
                    src_img = load_uploaded_image(file)
                    
                    if mask_method_batch == "HSV (Saturation)":
                        res = normalize_stain_reinhard_hsv_final(src_img, target_img, src_sat_thresh=batch_threshold, target_sat_thresh=batch_threshold)
                    else:
                        res = normalize_stain_reinhard_custom(src_img, target_img, src_thresh=batch_threshold, target_thresh=batch_threshold)
                    
                    is_success, buffer = cv2.imencode(".png", res)
                    if is_success:
                        original_name, _ = os.path.splitext(file.name)
                        zip_file.writestr(f"{original_name}_normalized.png", buffer.tobytes())
                    
                    progress_bar.progress((i + 1) / len(source_files))
                
                status_text.success("🎉 Batch Render abgeschlossen!")
                
            st.download_button(
                label="💾 Download Normalized Images (.zip)",
                data=zip_buffer.getvalue(),
                file_name="normalized_batch.zip",
                mime="application/zip"
            )

# ==========================================
# TAB 3: VIDEO ANALYSIS & INDIVIDUAL AUTO-SPLICER
# ==========================================
with tab_video:
    st.header("Video Auto-Splicer & Individual Grading")
    st.info("💡 **Workflow:** 1. Video analysieren & zerschneiden -> 2. Jede Szene individuell einstellen -> 3. Rendern & Downloaden.")
    
    col_vid1, col_vid2 = st.columns(2)
    with col_vid1:
        # WICHTIG: on_change löscht den Speicher, wenn ein neues Video kommt!
        video_file = st.file_uploader("Upload WSI Video (MP4)", type=["mp4", "avi", "mov"], on_change=reset_vid_state)
    with col_vid2:
        vid_target_file = st.file_uploader("Upload Target (Reference)", type=["jpg", "png", "tif"], key="trg_vid")

    if video_file and vid_target_file:
        target_img = load_uploaded_image(vid_target_file)
        
        # ---------------------------------------------------------
        # SCHRITT 1: ANALYSE & CUT DETECTION
        # ---------------------------------------------------------
        if st.session_state.vid_step == 1:
            st.markdown("### 🔍 Step 1: Global Cut Detection Settings")
            scene_thresh = st.slider("Scene Cut Sensitivity", 10.0, 100.0, 43.5, step=0.5)
            
            if st.button("✂️ Analyze Video & Extract Scenes"):
                tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile_in.write(video_file.read())
                
                cap = cv2.VideoCapture(tfile_in.name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                clip_idx = 1
                temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
                
                prev_gray = None
                cooldown = 0
                frame_idx = 0
                current_thumbnail = None
                
                vid_progress = st.progress(0)
                status = st.empty()
                status.info("Analysiere Video und extrahiere Roh-Clips...")
                
                # Leere den Speicher für die neuen Szenen
                st.session_state.vid_scenes = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    is_cut = False
                    
                    if cooldown > 0:
                        cooldown -= 1
                    elif prev_gray is not None:
                        diff = np.mean(cv2.absdiff(gray, prev_gray))
                        if diff > scene_thresh:
                            is_cut = True
                            cooldown = 15
                    prev_gray = gray
                    
                    if is_cut or frame_idx == 0:
                        current_thumbnail = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if is_cut:
                            out.release()
                            # Speichere Daten der fertigen Szene im Session State
                            st.session_state.vid_scenes.append({
                                'id': clip_idx,
                                'raw_path': temp_out.name,
                                'thumb': current_thumbnail_prev
                            })
                            clip_idx += 1
                            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                            out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
                            
                    current_thumbnail_prev = current_thumbnail
                    out.write(frame) # Wir schreiben das UNBEARBEITETE Originalbild
                    frame_idx += 1
                    
                # Letzte Szene speichern
                out.release()
                st.session_state.vid_scenes.append({
                    'id': clip_idx,
                    'raw_path': temp_out.name,
                    'thumb': current_thumbnail_prev
                })
                cap.release()
                
                # Gehe zu Schritt 2 und lade das UI neu
                st.session_state.vid_step = 2
                st.rerun()

        # ---------------------------------------------------------
        # SCHRITT 2: INDIVIDUAL GRADING & RENDER
        # ---------------------------------------------------------
        elif st.session_state.vid_step == 2:
            st.success(f"✅ Analyse abgeschlossen! {len(st.session_state.vid_scenes)} Szenen extrahiert.")
            st.markdown("### 🎛️ Step 2: Individual Scene Grading")
            
            # Globale Masken-Methode
            mask_method_vid = st.radio("Globale Masking Method für alle Clips", ["HSV (Saturation)", "Luma (Grayscale)"])
            
            # Dictionary zum Speichern der individuellen Slider-Werte
            scene_thresholds = {}
            
            # Baue ein schickes Grid für die Szenen (2 pro Reihe)
            cols = st.columns(2)
            for i, scene in enumerate(st.session_state.vid_scenes):
                with cols[i % 2]:
                    st.image(scene['thumb'], caption=f"Scene {scene['id']}", use_container_width=True)
                    # Jeder Clip bekommt seinen EIGENEN Slider!
                    if mask_method_vid == "HSV (Saturation)":
                        val = st.slider(f"HSV Threshold (Scene {scene['id']})", 0, 100, 15, key=f"sl_{scene['id']}")
                    else:
                        val = st.slider(f"Luma Threshold (Scene {scene['id']})", 0, 255, 210, key=f"sl_{scene['id']}")
                    scene_thresholds[scene['id']] = val
                    st.divider()

            # --- FINALE RENDER SCHLEIFE ---
            if st.button("🚀 Render All Scenes & Create ZIP", use_container_width=True):
                zip_buffer_vid = io.BytesIO()
                with zipfile.ZipFile(zip_buffer_vid, "w", zipfile.ZIP_DEFLATED) as zip_file_vid:
                    
                    render_bar = st.progress(0)
                    render_status = st.empty()
                    
                    for idx, scene in enumerate(st.session_state.vid_scenes):
                        render_status.text(f"Rendere Scene {scene['id']} von {len(st.session_state.vid_scenes)}...")
                        
                        # Roh-Video der Szene laden
                        cap_scene = cv2.VideoCapture(scene['raw_path'])
                        fps = cap_scene.get(cv2.CAP_PROP_FPS)
                        width = int(cap_scene.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap_scene.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        
                        # Temporäres Output-Video für diese Szene
                        temp_graded = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        out_scene = cv2.VideoWriter(temp_graded.name, fourcc, fps, (width, height))
                        
                        # Individuellen Threshold abrufen
                        thresh = scene_thresholds[scene['id']]
                        
                        while cap_scene.isOpened():
                            ret, frame = cap_scene.read()
                            if not ret: break
                            
                            # Grading mit dem spezifischen Wert!
                            try:
                                if mask_method_vid == "HSV (Saturation)":
                                    norm = normalize_stain_reinhard_hsv_final(frame, target_img, src_sat_thresh=thresh, target_sat_thresh=thresh)
                                else:
                                    norm = normalize_stain_reinhard_custom(frame, target_img, src_thresh=thresh, target_thresh=thresh)
                            except:
                                norm = frame
                                
                            out_scene.write(norm)
                            
                        cap_scene.release()
                        out_scene.release()
                        
                        # Das fertig gegradete Video ins ZIP packen
                        with open(temp_graded.name, "rb") as f:
                            zip_file_vid.writestr(f"graded_scene_{scene['id']:03d}.mp4", f.read())
                            
                        render_bar.progress((idx + 1) / len(st.session_state.vid_scenes))
                        
                render_status.success("🎉 Alle Clips gerendert und verpackt!")
                
                # Finaler Download Button
                st.download_button(
                    label="💾 Download Graded Scenes (.zip)",
                    data=zip_buffer_vid.getvalue(),
                    file_name="individually_graded_scenes.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                
            if st.button("🔄 Neustart (Anderes Video analysieren)"):
                reset_vid_state()
                st.rerun()