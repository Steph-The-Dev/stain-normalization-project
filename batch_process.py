import os
import cv2

# Importiere dein Modul
from src.reinhard import normalize_stain_reinhard_hsv_final

def run_batch_normalization(input_dir, output_dir, target_image_path, sat_thresh=15):
    """
    Nimmt alle Bilder aus input_dir, normalisiert sie auf das target_image 
    und speichert sie in output_dir.
    """
    print(f"🚀 Starte Batch-Processing...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(target_image_path):
        print("❌ Fehler: Target-Bild nicht gefunden!")
        return
        
    target_img = cv2.imread(target_image_path)
    
    # =========================================================
    # NEU: Der robuste Media-Scanner (Die Whitelist)
    # =========================================================
    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    
    source_files = []
    # Scanne alle Dateien im Input-Ordner
    if os.path.exists(input_dir):
        for file in os.listdir(input_dir):
            # Prüfe, ob die Datei (in Kleinbuchstaben) mit einer erlaubten Endung aufhört
            if file.lower().endswith(VALID_EXTENSIONS):
                source_files.append(os.path.join(input_dir, file))
                
    if not source_files:
        print(f"⚠️ Keine passenden Bilder (wie {VALID_EXTENSIONS}) im Ordner gefunden.")
        return

    print(f"📸 {len(source_files)} Bilder zum Verarbeiten gefunden.\n")

    # --- Die Render-Schleife ---
    for idx, file_path in enumerate(source_files, 1):
        filename = os.path.basename(file_path)
        print(f"⏳ Verarbeite [{idx}/{len(source_files)}]: {filename} ...", end="", flush=True)
        
        # Source laden (-1 als Flag bei imread stellt sicher, dass auch 16-bit TIFs gelesen werden)
        src_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if src_img is None:
            print(" ❌ Fehler beim Lesen (übersprungen).")
            continue
            
        # Falls es ein 16-bit TIF ist, für OpenCV (Reinhard) auf 8-bit runterskalieren
        if src_img.dtype == 'uint16':
            src_img = (src_img / 256).astype('uint8')
            
        try:
            # Normalisierung anwenden
            result_img = normalize_stain_reinhard_hsv_final(
                src_img, target_img, 
                src_sat_thresh=sat_thresh, 
                target_sat_thresh=sat_thresh
            )
            
            # Immer als hochwertiges Format (z.B. PNG oder TIF) speichern, um Kompressionsartefakte zu vermeiden
            name, _ = os.path.splitext(filename)
            save_path = os.path.join(output_dir, f"{name}_normalized.tif") 
            
            cv2.imwrite(save_path, result_img)
            print(" ✅ Fertig.")
            
        except Exception as e:
            print(f" ❌ Fehler: {e}")

    print(f"\n🎉 Batch-Processing abgeschlossen! Output: {output_dir}")

if __name__ == "__main__":
    INPUT_FOLDER = "data/raw/batch_input"
    OUTPUT_FOLDER = "data/processed/batch_output"
    TARGET_IMAGE = "data/raw/target.tif" 
    
    run_batch_normalization(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_IMAGE, sat_thresh=55)