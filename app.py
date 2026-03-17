import streamlit as st
import cv2
import numpy as np
import os
import tempfile

# ==========================================
# HILFSFUNKTION FÜR TIMECODES
# ==========================================
def format_timecode(seconds):
    """Wandelt Sekunden in ein lesbares Timecode-Format (MM:SS:MMM) um."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{mins:02d}:{secs:02d}:{ms:03d}"

def generate_fast_rgb_parade(image_bgr, scope_width=256, scope_height=400):
    """
    Rendert eine RGB-Parade als reines Pixel-Array (2D Histogramm).
    Mit logarithmischer Kompression für maximale Sichtbarkeit der Waveforms!
    """
    # 1. Bild verkleinern
    img = cv2.resize(image_bgr, (scope_width, 128))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Schwarze Leinwand (nutzen erst float32 für präzise Mathematik)
    parade = np.zeros((scope_height, scope_width * 3, 3), dtype=np.float32)
    
    for i in range(3): # 0=Rot, 1=Grün, 2=Blau
        channel = img_rgb[:, :, i]
        chan_scope = np.zeros((256, scope_width), dtype=np.float32)
        
        # 2. 2D Histogramm: Zähle Helligkeitswerte pro Spalte
        for x in range(scope_width):
            col_data = channel[:, x]
            hist = np.bincount(col_data, minlength=256)
            chan_scope[:, x] = hist[:256]
            
        # --- DER FIX: LOGARITHMISCHE KOMPRESSION ---
        # np.log1p (Logarithmus + 1) verhindert Fehler bei 0 und boostet schwache Signale enorm!
        chan_scope = np.log1p(chan_scope)
        
        # 3. Normalisieren auf 0.0 bis 1.0
        max_val = chan_scope.max()
        if max_val > 0:
            chan_scope = chan_scope / max_val
            
        # 4. Gamma-Kurve & Gain-Boost (Macht die Pixel richtig schön hell und "leuchtend")
        chan_scope = np.power(chan_scope, 0.6) * 255  # Gamma
        chan_scope = np.clip(chan_scope * 1.5, 0, 255) # Gain (1.5x Helligkeit)
        
        # 5. Vertikal spiegeln (Weiß/255 soll oben sein)
        chan_scope = np.flipud(chan_scope)
        chan_scope = cv2.resize(chan_scope, (scope_width, scope_height))
        
        # 6. In die Parade einfügen
        x_offset = i * scope_width
        parade[:, x_offset:x_offset+scope_width, i] = chan_scope
        
    # Am Ende zurück in 8-Bit Bilddaten umwandeln
    return parade.astype(np.uint8)
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

# Hilfsfunktion für den Bild-Upload (mit EOF-Fix)
def load_uploaded_image(uploaded_file):
    uploaded_file.seek(0)  # <--- WICHTIG: Spult die Datei zurück auf Anfang!
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

# --- HEADER ---
st.title("🔬 Histological Stain Normalization Suite")
st.markdown("**Powered by Reinhard Method & Smart Tissue Masking (HSV/Luma)**")

# --- UI TABS ---
tab_single, tab_batch, tab_video = st.tabs(["📷 Single Image", "📂 Batch Processing", "🎬 Video Analysis"])

# ==========================================
# HILFSFUNKTION FÜR ECHTZEIT-SLIDER (PROXY)
# (Füge das einfach oben im Tab 1 Bereich ein)
# ==========================================
def create_ui_proxy(image_bgr, max_height=400):
    """
    Skaliert Bilder für das Streamlit-UI herunter. 
    Reduziert die Rechenzeit von 500ms auf <10ms -> Flüssiger Slider!
    """
    h, w = image_bgr.shape[:2]
    if h > max_height:
        ratio = max_height / h
        return cv2.resize(image_bgr, (int(w * ratio), max_height))
    return image_bgr

# ==========================================
# TAB 1: SINGLE IMAGE (Pro Grading Layout)
# ==========================================
with tab_single:
    st.header("Single Image Look Dev")
    
    # --- MEDIA POOL (Oben) ---
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        source_file = st.file_uploader("Upload Source", type=["jpg", "png", "tif"], key="src_single")
    with col_up2:
        target_file = st.file_uploader("Upload Target (Reference)", type=["jpg", "png", "tif"], key="trg_single")

    st.divider()

    if source_file and target_file:
        raw_src = load_uploaded_image(source_file)
        raw_trg = load_uploaded_image(target_file)

        # Proxys extrem klein rechnen (max 300px Höhe), das verzehnfacht die Rechengeschwindigkeit!
        src_proxy = create_ui_proxy(raw_src, max_height=300)
        trg_proxy = create_ui_proxy(raw_trg, max_height=300)

        # ========================================================
        # ISOLIERTES LOOK DEV FRAGMENT
        # ========================================================
        @st.fragment
        def look_dev_panel():
            st.markdown("### 🎛️ Grading Controls")
            
            col_set1, col_set2, col_set3, col_set4 = st.columns([1, 1.5, 1.5, 1])
            with col_set1:
                mask_method_single = st.radio("Masking Method", ["HSV (Saturation)", "Luma (Grayscale)"], key="method_single")
            with col_set2:
                if mask_method_single == "HSV (Saturation)":
                    threshold_single = st.slider("Mask Threshold", 0, 100, 15, key="slider_single_hsv")
                else:
                    threshold_single = st.slider("Mask Threshold", 0, 255, 210, key="slider_single_luma")
            with col_set3:
                luma_blend_single = st.slider("Luma Preservation (Contrast)", 0.0, 1.0, 0.2, step=0.05, key="blend_single")
            with col_set4:
                show_scopes_single = st.toggle("📊 Show RGB Parades", value=True, key="scope_single")

            # --- SCHNELLE BERECHNUNG (Auf dem Proxy) ---
            if mask_method_single == "HSV (Saturation)":
                res_proxy = normalize_stain_reinhard_hsv_final(src_proxy, trg_proxy, src_sat_thresh=threshold_single, target_sat_thresh=threshold_single, luma_blend=luma_blend_single)
            else:
                res_proxy = normalize_stain_reinhard_custom(src_proxy, trg_proxy, src_thresh=threshold_single, target_thresh=threshold_single, luma_blend=luma_blend_single)

            st.markdown("### 📺 Grading Monitor")
            
            img_width = 350
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("**1. Source**")
                st.image(cv2.cvtColor(src_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                if show_scopes_single: st.image(generate_fast_rgb_parade(src_proxy), width=img_width)

            with c2:
                st.markdown("**2. Target**")
                st.image(cv2.cvtColor(trg_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                if show_scopes_single: st.image(generate_fast_rgb_parade(trg_proxy), width=img_width)

            with c3:
                st.markdown("**3. Result Preview**")
                st.image(cv2.cvtColor(res_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                if show_scopes_single: st.image(generate_fast_rgb_parade(res_proxy), width=img_width)

            st.divider()

            # --- NEU: FULL RESOLUTION EXPORT ---
            st.markdown("### 💾 Export Master Image")
            
            col_render, col_download = st.columns(2)
            
            with col_render:
                if st.button("🚀 Render High-Res Image", use_container_width=True):
                    with st.spinner("Berechne volle Auflösung..."):
                        # HIER nutzen wir die raw_src und raw_trg (die großen Originale!)
                        if mask_method_single == "HSV (Saturation)":
                            res_full = normalize_stain_reinhard_hsv_final(raw_src, raw_trg, src_sat_thresh=threshold_single, target_sat_thresh=threshold_single, luma_blend=luma_blend_single)
                        else:
                            res_full = normalize_stain_reinhard_custom(raw_src, raw_trg, src_thresh=threshold_single, target_thresh=threshold_single, luma_blend=luma_blend_single)
                        
                        # In PNG umwandeln für verlustfreien Export
                        is_success, buffer = cv2.imencode(".png", res_full)
                        if is_success:
                            # Speichere das fertige Bild im Session State
                            st.session_state['single_download_ready'] = buffer.tobytes()

            with col_download:
                # Zeige den Download-Button nur an, wenn gerendert wurde
                if 'single_download_ready' in st.session_state:
                    st.download_button(
                        label="⬇️ Download .PNG",
                        data=st.session_state['single_download_ready'],
                        file_name="normalized_master.png",
                        mime="image/png",
                        use_container_width=True
                    )

        # Fragment-Funktion aufrufen
        look_dev_panel()

# ==========================================
# TAB 2: BATCH PROCESSING (Mit Pro Layout & Fragment)
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
        
    if source_files and batch_target_file:
        raw_trg = load_uploaded_image(batch_target_file)
        raw_src = load_uploaded_image(source_files[0]) # Das erste Bild für die Vorschau
        
        # PROXYS für das Look Dev (macht den Slider flüssig)
        trg_proxy = create_ui_proxy(raw_trg, max_height=300)
        src_proxy = create_ui_proxy(raw_src, max_height=300)
        
        # ========================================================
        # ISOLIERTES LOOK DEV FRAGMENT (TAB 2)
        # ========================================================
        @st.fragment
        def batch_look_dev_panel():
            st.markdown("### 👁️ Look Dev: Live-Vorschau (Erstes Bild)")
            
            # Wir brauchen 4 Spalten für den neuen Regler
            col_bset1, col_bset2, col_bset3, col_bset4 = st.columns([1, 1.5, 1.5, 1])
            with col_bset1:
                method = st.radio("Masking Method", ["HSV (Saturation)", "Luma (Grayscale)"], key="method_batch")
            with col_bset2:
                if method == "HSV (Saturation)":
                    thresh = st.slider("Mask Threshold", 0, 100, 15, key="slider_batch_hsv")
                else:
                    thresh = st.slider("Mask Threshold", 0, 255, 210, key="slider_batch_luma")
            with col_bset3:
                # --- NEU: DER LUMINANCE BLEND FÜR DEN BATCH ---
                luma_blend_batch = st.slider("Luma Preservation", 0.0, 1.0, 0.2, step=0.05, key="blend_batch")
            with col_bset4:
                show_scopes = st.toggle("📊 Show RGB Parades", value=True, key="scope_batch")
                
            # Schnelle Berechnung auf dem Proxy (mit neuem Luma Blend)
            if method == "HSV (Saturation)":
                res_proxy = normalize_stain_reinhard_hsv_final(src_proxy, trg_proxy, src_sat_thresh=thresh, target_sat_thresh=thresh, luma_blend=luma_blend_batch)
            else:
                res_proxy = normalize_stain_reinhard_custom(src_proxy, trg_proxy, src_thresh=thresh, target_thresh=thresh, luma_blend=luma_blend_batch)
                
            img_width = 350
            c1, c2, c3 = st.columns(3)
            
            with c1:
                # FIX: Uniforme Überschrift verhindert das Verrutschen nach unten!
                st.markdown("**1. Source**")
                st.image(cv2.cvtColor(src_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                st.caption(f"File: {source_files[0].name}") # Dateiname sicher unter dem Bild
                if show_scopes: st.image(generate_fast_rgb_parade(src_proxy), width=img_width)
                
            with c2:
                st.markdown("**2. Target**")
                st.image(cv2.cvtColor(trg_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                if show_scopes: st.image(generate_fast_rgb_parade(trg_proxy), width=img_width)
                
            with c3:
                st.markdown("**3. Result Preview**")
                st.image(cv2.cvtColor(res_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                if show_scopes: st.image(generate_fast_rgb_parade(res_proxy), width=img_width)
                
            st.divider()
            
            # --- DER RENDER PROZESS ---
            if st.button("🚀 Start Full Batch Render (Apply to all images)", use_container_width=True):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file in enumerate(source_files):
                        status_text.text(f"Verarbeite Master: {file.name} ({i+1}/{len(source_files)})")
                        
                        full_src = load_uploaded_image(file)
                        
                        # Luma Blend im finalen Render anwenden!
                        if method == "HSV (Saturation)":
                            res = normalize_stain_reinhard_hsv_final(full_src, raw_trg, src_sat_thresh=thresh, target_sat_thresh=thresh, luma_blend=luma_blend_batch)
                        else:
                            res = normalize_stain_reinhard_custom(full_src, raw_trg, src_thresh=thresh, target_thresh=thresh, luma_blend=luma_blend_batch)
                        
                        is_success, buffer = cv2.imencode(".png", res)
                        if is_success:
                            original_name, _ = os.path.splitext(file.name)
                            zip_file.writestr(f"{original_name}_normalized.png", buffer.tobytes())
                        
                        progress_bar.progress((i + 1) / len(source_files))
                    
                    status_text.success("🎉 Full Resolution Batch Render abgeschlossen!")
                    
                st.download_button(
                    label="💾 Download Master Images (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name="normalized_batch.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                
        batch_look_dev_panel()

# ==========================================
# TAB 3: VIDEO ANALYSIS & INDIVIDUAL AUTO-SPLICER
# ==========================================
with tab_video:
    st.header("Video Auto-Splicer & Individual Grading")
    st.info("💡 **Workflow:** 1. Video analysieren & zerschneiden -> 2. Jede Szene individuell einstellen -> 3. Rendern.")
    
    col_vid1, col_vid2 = st.columns(2)
    with col_vid1:
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
                if fps == 0 or np.isnan(fps): fps = 25.0 # Fallback-Sicherheit
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                clip_idx = 1
                temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
                
                prev_gray = None
                cooldown = 0
                frame_idx = 0
                scene_start_frame = 0  # <--- NEU: Merkt sich den Start-Frame
                current_thumbnail = None
                current_thumbnail_prev = None
                
                vid_progress = st.progress(0)
                status = st.empty()
                status.info("Analysiere Video und extrahiere Roh-Clips...")
                
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
                            # NEU: Wir speichern Timecodes im Session State
                            st.session_state.vid_scenes.append({
                                'id': clip_idx,
                                'raw_path': temp_out.name,
                                'thumb': current_thumbnail_prev,
                                'start_time': scene_start_frame / fps,
                                'end_time': (frame_idx - 1) / fps
                            })
                            clip_idx += 1
                            scene_start_frame = frame_idx
                            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                            out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
                            
                    current_thumbnail_prev = current_thumbnail
                    out.write(frame)
                    frame_idx += 1
                    
                # Letzte Szene speichern
                out.release()
                if current_thumbnail_prev is not None:
                    st.session_state.vid_scenes.append({
                        'id': clip_idx,
                        'raw_path': temp_out.name,
                        'thumb': current_thumbnail_prev,
                        'start_time': scene_start_frame / fps,
                        'end_time': frame_idx / fps
                    })
                cap.release()
                
                st.session_state.vid_step = 2
                st.rerun()

        # ---------------------------------------------------------
        # SCHRITT 2: INDIVIDUAL GRADING (Mit Sticky Target Reference)
        # ---------------------------------------------------------
        elif st.session_state.vid_step == 2:
            st.success(f"✅ Analyse abgeschlossen! {len(st.session_state.vid_scenes)} Szenen extrahiert.")
            
            trg_proxy = create_ui_proxy(target_img, max_height=300)
            
            @st.fragment
            def video_look_dev_panel():
                st.markdown("### 🎛️ Step 2: Individual Scene Look Dev")
                show_scopes = st.toggle("📊 Show RGB Parades for all Scenes", value=True, key="vid_scopes")
                st.divider()
                
                # --- DAS NEUE LAYOUT ---
                col_main, col_ref = st.columns([2.5, 1], gap="large")
                
                # ---------------------------------------------------------
                # RECHTE SPALTE: Der statische Reference Monitor
                # ---------------------------------------------------------
                with col_ref:
                    st.markdown("#### 🎯 Master Target")
                    # Wir nutzen das rohe Target-Bild für maximale Qualität, ohne HTML-Rahmen!
                    st.image(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.info("💡 Scrolle durch die Timeline auf der linken Seite. Dieses Target-Bild bleibt immer hier stehen.")

                # ---------------------------------------------------------
                # LINKE SPALTE: Die scrollbare Timeline
                # ---------------------------------------------------------
                with col_main:
                    scene_settings = {}
                    img_width = 280 
                    
                    # --- DER MAGIC TRICK: Ein scrollbarer Container ---
                    with st.container(height=750):
                        for scene in st.session_state.vid_scenes:
                            tc_start = format_timecode(scene['start_time'])
                            tc_end = format_timecode(scene['end_time'])
                            st.markdown(f"#### 🎬 Scene {scene['id']} &nbsp;&nbsp;|&nbsp;&nbsp; ⏱️ `{tc_start}` - `{tc_end}`")
                            
                            col_controls, col_src, col_res = st.columns([1, 1.2, 1.2])
                            
                            with col_controls:
                                method = st.radio("Method", ["HSV (Sat)", "Luma (Gray)"], key=f"method_{scene['id']}")
                                if method == "HSV (Sat)":
                                    thresh = st.slider("Mask Thresh", 0, 100, 15, key=f"thresh_{scene['id']}")
                                else:
                                    thresh = st.slider("Mask Thresh", 0, 255, 210, key=f"thresh_{scene['id']}")
                                
                                luma_blend_scene = st.slider("Luma Preserve", 0.0, 1.0, 0.2, step=0.05, key=f"blend_{scene['id']}")
                                scene_settings[scene['id']] = {'method': method, 'thresh': thresh, 'blend': luma_blend_scene}

                            thumb_bgr = cv2.cvtColor(scene['thumb'], cv2.COLOR_RGB2BGR)
                            src_proxy = create_ui_proxy(thumb_bgr, max_height=300)
                            
                            try:
                                if method == "HSV (Sat)":
                                    res_proxy = normalize_stain_reinhard_hsv_final(src_proxy, trg_proxy, src_sat_thresh=thresh, target_sat_thresh=thresh, luma_blend=luma_blend_scene)
                                else:
                                    res_proxy = normalize_stain_reinhard_custom(src_proxy, trg_proxy, src_thresh=thresh, target_thresh=thresh, luma_blend=luma_blend_scene)
                            except:
                                res_proxy = src_proxy 

                            with col_src:
                                st.markdown("**Source**")
                                st.image(cv2.cvtColor(src_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                                if show_scopes: st.image(generate_fast_rgb_parade(src_proxy), width=img_width)
                                
                            with col_res:
                                st.markdown("**Preview**")
                                st.image(cv2.cvtColor(res_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                                if show_scopes: st.image(generate_fast_rgb_parade(res_proxy), width=img_width)
                                
                            st.divider()

                # ---------------------------------------------------------
                # FINALE RENDER SCHLEIFE (Unterhalb der Spalten)
                # ---------------------------------------------------------
                st.markdown("### 💾 Export Master Video")
                if st.button("🚀 Render Master ZIP (Apply all Settings)", use_container_width=True):
                    zip_buffer_vid = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer_vid, "w", zipfile.ZIP_DEFLATED) as zip_file_vid:
                        
                        render_bar = st.progress(0)
                        render_status = st.empty()
                        
                        for idx, scene in enumerate(st.session_state.vid_scenes):
                            render_status.text(f"Rendere Scene {scene['id']} von {len(st.session_state.vid_scenes)} (Full Resolution)...")
                            
                            cap_scene = cv2.VideoCapture(scene['raw_path'])
                            fps = cap_scene.get(cv2.CAP_PROP_FPS)
                            width = int(cap_scene.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap_scene.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            
                            temp_graded = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                            out_scene = cv2.VideoWriter(temp_graded.name, fourcc, fps, (width, height))
                            
                            c_method = scene_settings[scene['id']]['method']
                            c_thresh = scene_settings[scene['id']]['thresh']
                            c_blend = scene_settings[scene['id']]['blend']
                            
                            while cap_scene.isOpened():
                                ret, frame = cap_scene.read()
                                if not ret: break
                                
                                try:
                                    if c_method == "HSV (Sat)":
                                        norm = normalize_stain_reinhard_hsv_final(frame, target_img, src_sat_thresh=c_thresh, target_sat_thresh=c_thresh, luma_blend=c_blend)
                                    else:
                                        norm = normalize_stain_reinhard_custom(frame, target_img, src_thresh=c_thresh, target_thresh=c_thresh, luma_blend=c_blend)
                                except:
                                    norm = frame
                                    
                                out_scene.write(norm)
                                
                            cap_scene.release()
                            out_scene.release()
                            
                            with open(temp_graded.name, "rb") as f:
                                zip_file_vid.writestr(f"graded_scene_{scene['id']:03d}.mp4", f.read())
                                
                            render_bar.progress((idx + 1) / len(st.session_state.vid_scenes))
                            
                    render_status.success("🎉 Alle Clips in Master-Qualität gerendert und verpackt!")
                    
                    st.download_button(
                        label="💾 Download Spliced & Graded Scenes (.zip)",
                        data=zip_buffer_vid.getvalue(),
                        file_name="master_graded_scenes.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    
                if st.button("🔄 Neustart (Anderes Video analysieren)"):
                    reset_vid_state()
                    st.rerun()

            video_look_dev_panel()