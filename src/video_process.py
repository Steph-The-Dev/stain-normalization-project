"""
Video Processing Module for Stain Normalization with Scene Detection
"""

import cv2
import numpy as np
import os
import logging
import numpy.typing as npt
from src.reinhard import normalize_stain_reinhard_hsv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def process_video_with_scene_detection(
    input_path: str, 
    output_path: str, 
    target_image_path: str, 
    sat_thresh: int = 15, 
    scene_threshold: float = 43.5
) -> None:
    """
    Normalizes a video stream and detects scene cuts based on frame differencing.

    Parameters
    ----------
    input_path : str
        Path to the source video file.
    output_path : str
        Path to save the normalized video.
    target_image_path : str
        Path to the reference target image.
    sat_thresh : int, optional
        Saturation threshold for tissue masking, by default 15.
    scene_threshold : float, optional
        Mean pixel difference threshold for scene cut detection, by default 43.5.
    """
    logger.info("Initializing Video Processing Engine...")
    
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        logger.error(f"Target image not found: {target_image_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Metadata: {width}x{height}px | {fps} FPS | {total_frames} Frames total")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_frame_gray = None
    scene_cuts = []
    cooldown = 0
    frame_idx = 0

    logger.info("Starting Rendering & Cut Detection...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # A. Scene Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if cooldown > 0:
            cooldown -= 1
        elif prev_frame_gray is not None:
            diff = cv2.absdiff(gray, prev_frame_gray)
            mean_diff = np.mean(diff)

            if mean_diff > scene_threshold:
                timestamp = frame_idx / fps
                logger.info(f"Scene CUT detected at frame {frame_idx} | Diff: {mean_diff:.1f}")
                scene_cuts.append((frame_idx, timestamp))
                cooldown = 15 # 0.5s cooldown at 30fps
                
        prev_frame_gray = gray

        # B. Color Normalization
        try:
            norm_frame = normalize_stain_reinhard_hsv(
                frame, target_img, 
                src_sat_thresh=sat_thresh, 
                target_sat_thresh=sat_thresh
            )
        except Exception:
            # Fallback to original frame if math fails (e.g., pure white background)
            norm_frame = frame

        # C. Write Frame
        out.write(norm_frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            logger.info(f"Rendered: {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    
    logger.info(f"Rendering complete. Saved to: {output_path}")
    logger.info(f"Total scene cuts detected: {len(scene_cuts)}")
    for cut in scene_cuts:
        logger.info(f"  - Frame {cut[0]} ({cut[1]:.2f}s)")

if __name__ == "__main__":
    # Example usage for CLI testing
    INPUT_VIDEO = "data/raw/test_video.mp4"
    OUTPUT_VIDEO = "data/processed/normalized_video.mp4"
    TARGET_IMAGE = "data/raw/target.tif"
    
    if os.path.exists(INPUT_VIDEO):
        process_video_with_scene_detection(INPUT_VIDEO, OUTPUT_VIDEO, TARGET_IMAGE, sat_thresh=15)
    else:
        logger.warning(f"Test video not found at {INPUT_VIDEO}")
