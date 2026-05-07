import matplotlib.pyplot as plt
import cv2
import os

def create_linkedin_comparison(img_source_path, img_norm_path, output_path):
    if not os.path.exists(img_source_path):
        print(f"❌ Error: Source image not found at {img_source_path}")
        return
    if not os.path.exists(img_norm_path):
        print(f"❌ Error: Normalized image not found at {img_norm_path}")
        return

    img_src = cv2.cvtColor(cv2.imread(img_source_path), cv2.COLOR_BGR2RGB)
    img_norm = cv2.cvtColor(cv2.imread(img_norm_path), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=300)
    
    axes[0].imshow(img_src)
    axes[0].set_title("Original (Domain Shift)", fontsize=22, fontweight='bold', pad=15)
    axes[0].axis('off')

    axes[1].imshow(img_norm)
    axes[1].set_title("Normalized (Reinhard Method)", fontsize=22, fontweight='bold', pad=15)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Bild gespeichert unter: {output_path}")

# Dieser Block führt den Code aus, wenn du die Datei startest
if __name__ == "__main__":
    # Ersetze diese Pfade mit deinen echten Dateinamen
    SOURCE_IMG = "images/raw_camelyon_patch.jpg"
    NORM_IMG = "images/reinhard_camelyon_patch.jpg"
    OUTPUT = "linkedin_showcase_16x9.png"
    
    create_linkedin_comparison(SOURCE_IMG, NORM_IMG, OUTPUT)
