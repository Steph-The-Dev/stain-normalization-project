import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import zipfile
import io

from src.reinhard import normalize_stain_reinhard_hsv, normalize_stain_reinhard_luma
from src.normalizers import ReinhardNormalizer, MacenkoNormalizer, CUTStainNormalizer
from src.metrics import evaluate_normalization
from src.ui_utils import (
    format_timecode, 
    generate_fast_rgb_parade, 
    create_ui_proxy, 
    load_uploaded_image
)

# --- CONFIG ---
st.set_page_config(page_title="Stain Normalization Pro", page_icon="🔬", layout="wide")

# --- SESSION STATE SETUP (For 2-Step Video Workflow) ---
if 'vid_scenes' not in st.session_state:
    st.session_state.vid_scenes = []
if 'vid_step' not in st.session_state:
    st.session_state.vid_step = 1

def reset_vid_state():
    """Resets the video memory when a new video is uploaded."""
    st.session_state.vid_scenes = []
    st.session_state.vid_step = 1

# --- HEADER ---
st.title("🔬 Histological Stain Normalization Suite")
st.markdown("**Powered by Reinhard Method & Smart Tissue Masking (HSV/Luma)**")

# --- UI TABS ---
tab_single, tab_batch, tab_video = st.tabs(["📷 Single Image", "📂 Batch Processing", "🎬 Video Analysis"])

# ==========================================
# TAB 1: SINGLE IMAGE (Pro Grading Layout)
# ==========================================
with tab_single:
    st.header("Single Image Look Dev")
    
    # --- MEDIA POOL ---
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        source_file = st.file_uploader("Upload Source", type=["jpg", "png", "tif"], key="src_single")
    with col_up2:
        target_file = st.file_uploader("Upload Target (Reference)", type=["jpg", "png", "tif"], key="trg_single")

    use_demo = st.checkbox("🧪 Use Built-in Demo Samples (data/raw/source.tif & target.tif)", value=False, key="demo_single")

    raw_src, raw_trg = None, None
    if use_demo:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        demo_src_path = os.path.join(base_dir, "data", "raw", "source.tif")
        demo_trg_path = os.path.join(base_dir, "data", "raw", "target.tif")
        if os.path.exists(demo_src_path) and os.path.exists(demo_trg_path):
            raw_src = cv2.imread(demo_src_path)
            raw_trg = cv2.imread(demo_trg_path)
        else:
            st.warning("Built-in demo images not found in data/raw.")
    elif source_file and target_file:
        raw_src = load_uploaded_image(source_file)
        raw_trg = load_uploaded_image(target_file)

    st.divider()

    if raw_src is not None and raw_trg is not None:
        # Proxies (max 300px height) significantly boost UI responsiveness
        src_proxy = create_ui_proxy(raw_src, max_height=300)
        trg_proxy = create_ui_proxy(raw_trg, max_height=300)

        st.markdown("### 🎛️ Grading & Normalizer Model Controls")
        
        col_algo, col_ctrl, col_scope = st.columns([1.5, 3.5, 1])
        with col_algo:
            algo_choice = st.selectbox(
                "Normalization Model", 
                ["Reinhard (CIELAB)", "Macenko (SVD Optical Density)", "CUT (Deep Learning GAN)"],
                key="algo_single"
            )
        with col_scope:
            show_scopes_single = st.toggle("📊 Show RGB Parades", value=True, key="scope_single")

        # Model-Specific Dynamic Controls
        if "Macenko" in algo_choice:
            with col_ctrl:
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    macenko_thresh = st.slider("Tissue Mask Saturation Threshold", 0, 100, 15, key="macenko_thresh")
                with col_m2:
                    macenko_beta = st.slider("OD Background Cutoff (Beta)", 0.01, 0.40, 0.15, step=0.01, key="macenko_beta")
            
            norm = MacenkoNormalizer(saturation_threshold=macenko_thresh, beta=macenko_beta)
            res_proxy = norm.fit_transform(src_proxy, trg_proxy)

        elif "CUT" in algo_choice:
            with col_ctrl:
                col_cut1, col_cut2, col_cut3 = st.columns([2, 2, 1])
                with col_cut1:
                    cut_epochs = st.slider("Training Epochs", 1, 30, 10, key="cut_epochs_slider")
                with col_cut2:
                    cut_color_weight = st.slider("Stain Strength (Color Weight)", 1.0, 20.0, 5.0, step=0.5, key="cut_color_weight_slider")
                with col_cut3:
                    train_cut_btn = st.button("🏋️ Train CUT Model", width='stretch', key="btn_train_cut")

            if train_cut_btn or 'cut_normalizer' not in st.session_state:
                st.session_state.cut_normalizer = CUTStainNormalizer(
                    ngf=16, num_blocks=3, lr=3e-3, lambda_color=cut_color_weight
                )

            if train_cut_btn or not st.session_state.cut_normalizer.is_fitted:
                if train_cut_btn:
                    with st.spinner(f"Training CUT Generator for {cut_epochs} epochs with Stain Weight {cut_color_weight:.1f}..."):
                        st.session_state.cut_normalizer.fit(trg_proxy, source_image=src_proxy, num_epochs=cut_epochs, batch_size=1)
                    st.success("✅ CUT Generator successfully trained!")
                else:
                    st.info("ℹ️ **Neural Network Status:** CUT model is initialized. Click **'🏋️ Train CUT Model'** above to optimize generator weights on your target slide!")

            if st.session_state.cut_normalizer.is_fitted:
                res_proxy = st.session_state.cut_normalizer.transform(src_proxy)
            else:
                res_proxy = src_proxy.copy()

        else:
            with col_ctrl:
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    mask_method_single = st.radio("Masking Method", ["HSV (Saturation)", "Luma (Grayscale)"], key="method_single")
                with col_r2:
                    if mask_method_single == "HSV (Saturation)":
                        threshold_single = st.slider("Mask Threshold", 0, 100, 15, key="slider_single_hsv")
                    else:
                        threshold_single = st.slider("Mask Threshold", 0, 255, 210, key="slider_single_luma")
                with col_r3:
                    luma_blend_single = st.slider("Luma Preservation", 0.0, 1.0, 0.2, step=0.05, key="blend_single")

            mask_m = "hsv" if mask_method_single == "HSV (Saturation)" else "luma"
            if mask_m == "hsv":
                res_proxy = normalize_stain_reinhard_hsv(
                    src_proxy, trg_proxy, 
                    src_sat_thresh=threshold_single, 
                    target_sat_thresh=threshold_single, 
                    luma_blend=luma_blend_single
                )
            else:
                res_proxy = normalize_stain_reinhard_luma(
                    src_proxy, trg_proxy, 
                    src_thresh=threshold_single, 
                    target_thresh=threshold_single, 
                    luma_blend=luma_blend_single
                )

        # Quantitative evaluation metrics
        metrics = evaluate_normalization(src_proxy, res_proxy, trg_proxy)
        st.markdown("### 📊 Quantitative Quality Metrics (Guardrails)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SSIM (Morphology Preservation)", f"{metrics['ssim_structural_preservation']:.3f}")
        m2.metric("PSNR (dB)", f"{metrics['psnr_db']:.2f}")
        m3.metric("Color Delta L (Luminance)", f"{metrics['target_delta_L']:.2f}")
        m4.metric("Color Delta ab (Chromaticity)", f"{metrics['target_delta_ab']:.2f}")

        st.markdown("### 📺 Grading Monitor")
        
        img_width = 350
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("**1. Source**")
            st.image(cv2.cvtColor(src_proxy, cv2.COLOR_BGR2RGB), width=img_width)
            if show_scopes_single: 
                st.image(generate_fast_rgb_parade(src_proxy), width=img_width)

        with c2:
            st.markdown("**2. Target**")
            st.image(cv2.cvtColor(trg_proxy, cv2.COLOR_BGR2RGB), width=img_width)
            if show_scopes_single: 
                st.image(generate_fast_rgb_parade(trg_proxy), width=img_width)

        with c3:
            st.markdown("**3. Result Preview**")
            st.image(cv2.cvtColor(res_proxy, cv2.COLOR_BGR2RGB), width=img_width)
            if show_scopes_single: 
                st.image(generate_fast_rgb_parade(res_proxy), width=img_width)

        st.divider()

        # --- FULL RESOLUTION EXPORT ---
        st.markdown("### 💾 Export Master Image")
        
        col_render, col_download = st.columns(2)
        
        with col_render:
            if st.button("🚀 Render High-Res Image", width='stretch'):
                with st.spinner("Calculating full resolution..."):
                    if "Macenko" in algo_choice:
                        norm_full = MacenkoNormalizer(saturation_threshold=macenko_thresh, beta=macenko_beta)
                        res_full = norm_full.fit_transform(raw_src, raw_trg)
                    elif "CUT" in algo_choice:
                        if 'cut_normalizer' in st.session_state and st.session_state.cut_normalizer.is_fitted:
                            res_full = st.session_state.cut_normalizer.transform(raw_src)
                        else:
                            norm_full = CUTStainNormalizer(ngf=16, num_blocks=3, lr=3e-3)
                            res_full = norm_full.fit_transform(raw_src, raw_trg)
                    else:
                        mask_m = "hsv" if mask_method_single == "HSV (Saturation)" else "luma"
                        norm_full = ReinhardNormalizer(mask_method=mask_m, threshold=threshold_single)
                        res_full = norm_full.fit_transform(raw_src, raw_trg)
                    
                    is_success, buffer = cv2.imencode(".png", res_full)
                    if is_success:
                        st.session_state['single_download_ready'] = buffer.tobytes()

        with col_download:
            if 'single_download_ready' in st.session_state:
                st.download_button(
                    label="⬇️ Download .PNG",
                    data=st.session_state['single_download_ready'],
                    file_name="normalized_master.png",
                    mime="image/png",
                    width='stretch'
                )
    else:
        st.info("💡 **Getting Started:** Upload both a **Source** and **Target** image above, or check **'🧪 Use Built-in Demo Samples'** to display the normalizer controls and metrics immediately!")

# ==========================================
# TAB 2: BATCH PROCESSING
# ==========================================
with tab_batch:
    st.header("Cloud Batch Rendering")
    st.info("Upload multiple images. Adjust settings using the live preview of the first image.")
    
    col_batch_up1, col_batch_up2 = st.columns(2)
    with col_batch_up1:
        source_files = st.file_uploader("Upload Source Images (Select Multiple)", type=["jpg", "png", "tif"], accept_multiple_files=True, key="src_batch")
    with col_batch_up2:
        batch_target_file = st.file_uploader("Upload Target (Reference)", type=["jpg", "png", "tif"], key="trg_batch")
        
    if source_files and batch_target_file:
        raw_trg = load_uploaded_image(batch_target_file)
        raw_src = load_uploaded_image(source_files[0]) # Use first image for preview
        
        trg_proxy = create_ui_proxy(raw_trg, max_height=300)
        src_proxy = create_ui_proxy(raw_src, max_height=300)
        
        @st.fragment
        def batch_look_dev_panel():
            st.markdown("### 👁️ Look Dev: Live Preview (First Image)")
            
            col_bset1, col_bset2, col_bset3, col_bset4 = st.columns([1, 1.5, 1.5, 1])
            with col_bset1:
                method = st.radio("Masking Method", ["HSV (Saturation)", "Luma (Grayscale)"], key="method_batch")
            with col_bset2:
                if method == "HSV (Saturation)":
                    thresh = st.slider("Mask Threshold", 0, 100, 15, key="slider_batch_hsv")
                else:
                    thresh = st.slider("Mask Threshold", 0, 255, 210, key="slider_batch_luma")
            with col_bset3:
                luma_blend_batch = st.slider("Luma Preservation", 0.0, 1.0, 0.2, step=0.05, key="blend_batch")
            with col_bset4:
                show_scopes = st.toggle("📊 Show RGB Parades", value=True, key="scope_batch")
                
            if method == "HSV (Saturation)":
                res_proxy = normalize_stain_reinhard_hsv(
                    src_proxy, trg_proxy, 
                    src_sat_thresh=thresh, 
                    target_sat_thresh=thresh, 
                    luma_blend=luma_blend_batch
                )
            else:
                res_proxy = normalize_stain_reinhard_luma(
                    src_proxy, trg_proxy, 
                    src_thresh=thresh, 
                    target_thresh=thresh, 
                    luma_blend=luma_blend_batch
                )
                
            img_width = 350
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("**1. Source**")
                st.image(cv2.cvtColor(src_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                if show_scopes: 
                    st.image(generate_fast_rgb_parade(src_proxy), width=img_width)
                
            with c2:
                st.markdown("**2. Target**")
                st.image(cv2.cvtColor(trg_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                if show_scopes: 
                    st.image(generate_fast_rgb_parade(trg_proxy), width=img_width)
                
            with c3:
                st.markdown("**3. Result Preview**")
                st.image(cv2.cvtColor(res_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                if show_scopes: 
                    st.image(generate_fast_rgb_parade(res_proxy), width=img_width)
                
            st.divider()
            
            if st.button("🚀 Start Full Batch Render (Apply to all images)", width='stretch'):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file in enumerate(source_files):
                        status_text.text(f"Processing Master: {file.name} ({i+1}/{len(source_files)})")
                        full_src = load_uploaded_image(file)
                        
                        if method == "HSV (Saturation)":
                            res = normalize_stain_reinhard_hsv(
                                full_src, raw_trg, 
                                src_sat_thresh=thresh, 
                                target_sat_thresh=thresh, 
                                luma_blend=luma_blend_batch
                            )
                        else:
                            res = normalize_stain_reinhard_luma(
                                full_src, raw_trg, 
                                src_thresh=thresh, 
                                target_thresh=thresh, 
                                luma_blend=luma_blend_batch
                            )
                        
                        is_success, buffer = cv2.imencode(".png", res)
                        if is_success:
                            original_name, _ = os.path.splitext(file.name)
                            zip_file.writestr(f"{original_name}_normalized.png", buffer.tobytes())
                        
                        progress_bar.progress((i + 1) / len(source_files))
                    
                    status_text.success("🎉 Full Resolution Batch Render Complete!")
                    
                st.download_button(
                    label="💾 Download Master Images (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name="normalized_batch.zip",
                    mime="application/zip",
                    width='stretch'
                )
                
        batch_look_dev_panel()

# ==========================================
# TAB 3: VIDEO ANALYSIS & AUTO-SPLICER
# ==========================================
with tab_video:
    st.header("Video Auto-Splicer & Individual Grading")
    st.info("💡 **Workflow:** 1. Analyze video & split into scenes -> 2. Adjust each scene individually -> 3. Render.")
    
    col_vid1, col_vid2 = st.columns(2)
    with col_vid1:
        video_file = st.file_uploader("Upload WSI Video (MP4)", type=["mp4", "avi", "mov"], on_change=reset_vid_state)
    with col_vid2:
        vid_target_file = st.file_uploader("Upload Target (Reference)", type=["jpg", "png", "tif"], key="trg_vid")

    if video_file and vid_target_file:
        target_img = load_uploaded_image(vid_target_file)
        
        # --- STEP 1: ANALYSIS & CUT DETECTION ---
        if st.session_state.vid_step == 1:
            st.markdown("### 🔍 Step 1: Global Cut Detection Settings")
            scene_thresh = st.slider("Scene Cut Sensitivity", 10.0, 100.0, 43.5, step=0.5)
            
            if st.button("✂️ Analyze Video & Extract Scenes"):
                tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile_in.write(video_file.read())
                
                cap = cv2.VideoCapture(tfile_in.name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or np.isnan(fps): fps = 25.0
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                clip_idx = 1
                temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
                
                prev_gray = None
                cooldown = 0
                frame_idx = 0
                scene_start_frame = 0
                current_thumbnail = None
                current_thumbnail_prev = None
                
                vid_progress = st.progress(0)
                status = st.empty()
                status.info("Analyzing video and extracting raw clips...")
                
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

        # --- STEP 2: INDIVIDUAL GRADING ---
        elif st.session_state.vid_step == 2:
            st.success(f"✅ Analysis complete! {len(st.session_state.vid_scenes)} scenes extracted.")
            
            trg_proxy = create_ui_proxy(target_img, max_height=300)
            
            @st.fragment
            def video_look_dev_panel():
                st.markdown("### 🎛️ Step 2: Individual Scene Look Dev")
                show_scopes = st.toggle("📊 Show RGB Parades for all Scenes", value=True, key="vid_scopes")
                st.divider()
                
                col_main, col_ref = st.columns([2.5, 1], gap="large")
                
                with col_ref:
                    st.markdown("#### 🎯 Master Target")
                    st.image(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB), width='stretch')
                    if show_scopes:
                        st.image(generate_fast_rgb_parade(trg_proxy), width='stretch')
                    st.info("💡 Scroll through the timeline on the left. The target image remains fixed.")

                with col_main:
                    scene_settings = {}
                    img_width = 280 
                    
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
                                    res_proxy = normalize_stain_reinhard_hsv(
                                        src_proxy, trg_proxy, 
                                        src_sat_thresh=thresh, 
                                        target_sat_thresh=thresh, 
                                        luma_blend=luma_blend_scene
                                    )
                                else:
                                    res_proxy = normalize_stain_reinhard_luma(
                                        src_proxy, trg_proxy, 
                                        src_thresh=thresh, 
                                        target_thresh=thresh, 
                                        luma_blend=luma_blend_scene
                                    )
                            except:
                                res_proxy = src_proxy 

                            with col_src:
                                st.markdown("**Source**")
                                st.image(cv2.cvtColor(src_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                                if show_scopes: 
                                    st.image(generate_fast_rgb_parade(src_proxy), width=img_width)
                                
                            with col_res:
                                st.markdown("**Preview**")
                                st.image(cv2.cvtColor(res_proxy, cv2.COLOR_BGR2RGB), width=img_width)
                                if show_scopes: 
                                    st.image(generate_fast_rgb_parade(res_proxy), width=img_width)
                                
                            st.divider()

                # --- FINAL RENDER LOOP ---
                st.markdown("### 💾 Export Master Video")
                if st.button("🚀 Render Master ZIP (Apply all Settings)", width='stretch'):
                    zip_buffer_vid = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer_vid, "w", zipfile.ZIP_DEFLATED) as zip_file_vid:
                        
                        render_bar = st.progress(0)
                        render_status = st.empty()
                        
                        for idx, scene in enumerate(st.session_state.vid_scenes):
                            render_status.text(f"Rendering Scene {scene['id']} of {len(st.session_state.vid_scenes)} (Full Resolution)...")
                            
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
                                        norm = normalize_stain_reinhard_hsv(
                                            frame, target_img, 
                                            src_sat_thresh=c_thresh, 
                                            target_sat_thresh=c_thresh, 
                                            luma_blend=c_blend
                                        )
                                    else:
                                        norm = normalize_stain_reinhard_luma(
                                            frame, target_img, 
                                            src_thresh=c_thresh, 
                                            target_thresh=c_thresh, 
                                            luma_blend=c_blend
                                        )
                                except:
                                    norm = frame
                                    
                                out_scene.write(norm)
                                
                            cap_scene.release()
                            out_scene.release()
                            
                            with open(temp_graded.name, "rb") as f:
                                zip_file_vid.writestr(f"graded_scene_{scene['id']:03d}.mp4", f.read())
                                
                            render_bar.progress((idx + 1) / len(st.session_state.vid_scenes))
                            
                    render_status.success("🎉 All clips rendered in master quality!")
                    
                    st.download_button(
                        label="💾 Download Spliced & Graded Scenes (.zip)",
                        data=zip_buffer_vid.getvalue(),
                        file_name="master_graded_scenes.zip",
                        mime="application/zip",
                        width='stretch'
                    )
                    
                if st.button("🔄 Restart (Analyze another video)"):
                    reset_vid_state()
                    st.rerun()

            video_look_dev_panel()
