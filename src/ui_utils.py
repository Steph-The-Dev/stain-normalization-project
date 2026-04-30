"""
UI Utility functions for the Stain Normalization Streamlit App
"""

import cv2
import numpy as np
import numpy.typing as npt
import streamlit as st


def format_timecode(seconds: float) -> str:
    """
    Converts seconds into a readable timecode format (MM:SS:MMM).

    Parameters
    ----------
    seconds : float
        Time in seconds.

    Returns
    -------
    str
        Formatted timecode string.
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{mins:02d}:{secs:02d}:{ms:03d}"


def generate_fast_rgb_parade(
    image_bgr: npt.NDArray[np.uint8], 
    scope_width: int = 256, 
    scope_height: int = 400
) -> npt.NDArray[np.uint8]:
    """
    Renders an RGB Parade as a pure pixel array (2D Histogram).
    Uses logarithmic compression for maximum waveform visibility.

    Parameters
    ----------
    image_bgr : npt.NDArray[np.uint8]
        Input image in BGR format.
    scope_width : int, optional
        Width of each channel's scope, by default 256.
    scope_height : int, optional
        Height of the scope, by default 400.

    Returns
    -------
    npt.NDArray[np.uint8]
        The rendered RGB Parade image.
    """
    # 1. Resize for performance
    img = cv2.resize(image_bgr, (scope_width, 128))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Canvas (using float32 for precise math before clipping)
    parade = np.zeros((scope_height, scope_width * 3, 3), dtype=np.float32)
    
    for i in range(3):  # 0=Red, 1=Green, 2=Blue
        channel = img_rgb[:, :, i]
        chan_scope = np.zeros((256, scope_width), dtype=np.float32)
        
        # 2. 2D Histogram: Count brightness values per column
        for x in range(scope_width):
            col_data = channel[:, x]
            hist = np.bincount(col_data, minlength=256)
            chan_scope[:, x] = hist[:256]
            
        # Logarithmic compression to boost weak signals
        chan_scope = np.log1p(chan_scope)
        
        # 3. Normalize to 0.0 - 1.0
        max_val = chan_scope.max()
        if max_val > 0:
            chan_scope = chan_scope / max_val
            
        # 4. Gamma curve & Gain boost for "glowing" effect
        chan_scope = np.power(chan_scope, 0.6) * 255
        chan_scope = np.clip(chan_scope * 1.5, 0, 255)
        
        # 5. Flip vertically (white/255 at top)
        chan_scope = np.flipud(chan_scope)
        chan_scope = cv2.resize(chan_scope, (scope_width, scope_height))
        
        # 6. Insert into parade
        x_offset = i * scope_width
        parade[:, x_offset:x_offset+scope_width, i] = chan_scope
        
    return parade.astype(np.uint8)


def create_ui_proxy(image_bgr: npt.NDArray[np.uint8], max_height: int = 400) -> npt.NDArray[np.uint8]:
    """
    Scales images down for the Streamlit UI to ensure smooth slider performance.

    Parameters
    ----------
    image_bgr : npt.NDArray[np.uint8]
        High-res source image.
    max_height : int, optional
        Target height for the proxy, by default 400.

    Returns
    -------
    npt.NDArray[np.uint8]
        Scaled-down proxy image.
    """
    h, w = image_bgr.shape[:2]
    if h <= max_height:
        return image_bgr
    
    scale = max_height / h
    new_w = int(w * scale)
    return cv2.resize(image_bgr, (new_w, max_height), interpolation=cv2.INTER_AREA)


def load_uploaded_image(uploaded_file) -> npt.NDArray[np.uint8]:
    """
    Decodes an uploaded Streamlit file into an OpenCV BGR image.
    """
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)
