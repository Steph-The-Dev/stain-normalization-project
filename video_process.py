import cv2
import numpy as np
import os

# Importiere dein Modul
from src.reinhard import normalize_stain_reinhard_hsv_final

def process_video_with_scene_detection(input_path, output_path, target_image_path, sat_thresh=15, scene_threshold=43.5):
    print(f"🎬 Initialisiere Video-Engine...")
    
    # 1. Target-Bild (Referenz) laden
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        print("❌ Fehler: Target-Bild nicht gefunden!")
        return

    # 2. Video einlesen
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Fehler: Konnte Video nicht öffnen: {input_path}")
        return

    # Video-Metadaten auslesen
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📼 Metadaten: {width}x{height}px | {fps} FPS | {total_frames} Frames total")

    # 3. Video-Writer einrichten (Der Encoder für den Export)
    # 'mp4v' ist ein sehr verlässlicher und weit verbreiteter Codec in OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Variablen für die Scene Detection
    prev_frame_gray = None
    scene_cuts = []

    print("🚀 Starte Rendering & Cut Detection...\n")

    cooldown = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Ende des Videos erreicht

        # ==========================================
        # A. SCENE DETECTION (Schnitt-Erkennung)
        # ==========================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if cooldown > 0:
            cooldown -= 1  # Cooldown runterzählen
        elif prev_frame_gray is not None:
            diff = cv2.absdiff(gray, prev_frame_gray)
            mean_diff = np.mean(diff)

            if mean_diff > scene_threshold:
                timestamp = frame_idx / fps
                print(f"✂️ CUT bei Frame {frame_idx} | Diff: {mean_diff:.1f}")
                scene_cuts.append((frame_idx, timestamp))
                cooldown = 15  # Setze Cooldown auf 15 Frames (Halbe Sekunde "Blindheit")
                
        prev_frame_gray = gray

        # ==========================================
        # B. FARB-NORMALISIERUNG (Reinhard)
        # ==========================================
        try:
            norm_frame = normalize_stain_reinhard_hsv_final(
                frame, target_img, 
                src_sat_thresh=sat_thresh, 
                target_sat_thresh=sat_thresh
            )
        except Exception:
            # Fallback: Falls ein Frame z.B. nur aus Glas besteht und die Mathematik zickt,
            # schreiben wir einfach den Original-Frame ins Video, um Abstürze zu vermeiden.
            norm_frame = frame

        # ==========================================
        # C. FRAME IN NEUES VIDEO SCHREIBEN
        # ==========================================
        out.write(norm_frame)
        
        frame_idx += 1
        
        # Kleine Fortschrittsanzeige im Terminal (alle 50 Frames)
        if frame_idx % 50 == 0:
            print(f"   ⏳ Gerendert: {frame_idx}/{total_frames} Frames...")

    # 4. Aufräumen (Ressourcen freigeben)
    cap.release()
    out.release()
    
    print(f"\n🎉 Render abgeschlossen!")
    print(f"💾 Gespeichert unter: {output_path}")
    print(f"📊 Gefundene Szenen-Schnitte: {len(scene_cuts)}")
    for cut in scene_cuts:
        print(f"   - Frame {cut[0]} ({cut[1]:.2f}s)")

# --- Ausführung ---
if __name__ == "__main__":
    # Definiere deine Pfade
    INPUT_VIDEO = "data/raw/test_video.mp4"
    OUTPUT_VIDEO = "data/processed/normalized_video.mp4"
    TARGET_IMAGE = "data/raw/target.tif"  # Pass den Namen ggf. wieder auf .jpg an
    
    # Test-Aufruf
    if os.path.exists(INPUT_VIDEO):
        process_video_with_scene_detection(INPUT_VIDEO, OUTPUT_VIDEO, TARGET_IMAGE, sat_thresh=15)
    else:
        print(f"⚠️ Bitte lege ein Test-Video unter {INPUT_VIDEO} ab!")