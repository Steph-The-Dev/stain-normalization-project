"""
Batch Processing Module for Stain Normalization
"""

import os
import cv2
import logging
import numpy as np
import numpy.typing as npt
from src.reinhard import normalize_stain_reinhard_hsv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_batch_normalization(
    input_dir: str, 
    output_dir: str, 
    target_image_path: str, 
    sat_thresh: int = 15
) -> None:
    """
    Process all images in input_dir, normalize them against target_image,
    and save the results to output_dir.

    Parameters
    ----------
    input_dir : str
        Directory containing source images.
    output_dir : str
        Directory to save normalized images.
    target_image_path : str
        Path to the reference target image.
    sat_thresh : int, optional
        Saturation threshold for tissue masking, by default 15.
    """
    logger.info("Initializing Batch Processing...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(target_image_path):
        logger.error(f"Target image not found: {target_image_path}")
        return
        
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        logger.error(f"Failed to load target image: {target_image_path}")
        return
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    source_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ]
                
    if not source_files:
        logger.warning(f"No valid images found in {input_dir}. Supported: {valid_extensions}")
        return

    logger.info(f"Found {len(source_files)} images for processing.")

    for idx, file_path in enumerate(source_files, 1):
        filename = os.path.basename(file_path)
        logger.info(f"[{idx}/{len(source_files)}] Processing: {filename}")
        
        # Load source (-1 flag ensures 16-bit TIFs are read correctly)
        src_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if src_img is None:
            logger.error(f"Failed to read {filename}. Skipping.")
            continue
            
        # Downscale 16-bit TIF to 8-bit for Reinhard processing
        if src_img.dtype == 'uint16':
            src_img = (src_img / 256).astype('uint8')
            
        try:
            result_img = normalize_stain_reinhard_hsv(
                src_img, target_img, 
                src_sat_thresh=sat_thresh, 
                target_sat_thresh=sat_thresh
            )
            
            # Save as TIFF to avoid compression artifacts
            name, _ = os.path.splitext(filename)
            save_path = os.path.join(output_dir, f"{name}_normalized.tif") 
            
            cv2.imwrite(save_path, result_img)
            
        except Exception as e:
            logger.exception(f"Error processing {filename}: {e}")

    logger.info(f"Batch processing complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    # Example usage for CLI testing
    INPUT_FOLDER = "data/raw/batch_input"
    OUTPUT_FOLDER = "data/processed/batch_output"
    TARGET_IMAGE = "data/raw/target.tif" 
    
    run_batch_normalization(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_IMAGE, sat_thresh=55)
