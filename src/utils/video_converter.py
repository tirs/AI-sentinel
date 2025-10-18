"""
Video conversion utility for dashboard video playback support
Converts various video formats to H.264 MP4 for Streamlit compatibility
"""

import cv2
import numpy as np
import tempfile
import os
import hashlib
from pathlib import Path
from typing import Tuple, Optional
from io import BytesIO
import streamlit as st
from src.utils import get_logger

logger = get_logger(__name__)

# Persistent cache directory for converted videos
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache")) / "videos"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_video_properties(video_path: str) -> dict:
    """Get video properties without loading entire file"""
    cap = cv2.VideoCapture(video_path)
    props = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return props


def convert_video_to_h264_mp4(input_bytes: bytes, filename: str, max_width: int = 1280) -> Tuple[str, str]:
    """
    Convert any video format to H.264 MP4 for Streamlit compatibility

    Args:
        input_bytes: Raw video file bytes
        filename: Original filename (for format detection)
        max_width: Maximum output width (height scales proportionally)

    Returns:
        Tuple of (filepath to converted video, output_filename)
    """
    try:
        # Create cache filename based on input hash
        file_hash = hashlib.md5(input_bytes).hexdigest()
        cache_filename = f"{file_hash}.mp4"
        output_path = CACHE_DIR / cache_filename

        # Return if already cached
        if output_path.exists():
            logger.info(f"Using cached video: {cache_filename}")
            return str(output_path), cache_filename

        # Save input to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_input:
            tmp_input.write(input_bytes)
            input_path = tmp_input.name

        try:
            # Open video
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                raise ValueError(f"Cannot read video file: {filename}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps == 0 or width == 0 or height == 0:
                raise ValueError("Invalid video properties detected")

            # Calculate new dimensions (maintain aspect ratio)
            if width > max_width:
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                # Ensure even dimensions for codec
                new_height = new_height if new_height % 2 == 0 else new_height - 1
            else:
                new_width = width if width % 2 == 0 else width - 1
                new_height = height if height % 2 == 0 else height - 1

            # Define codec and create writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 compatible codec
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_width, new_height))

            if not out.isOpened():
                raise ValueError("Cannot create output video")

            # Process frames with progress tracking
            frame_idx = 0
            progress_bar = st.progress(0)
            status_text = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame
                if (frame.shape[1], frame.shape[0]) != (new_width, new_height):
                    frame = cv2.resize(frame, (new_width, new_height))

                out.write(frame)
                frame_idx += 1

                # Update progress every 10 frames
                if frame_idx % 10 == 0:
                    progress = min(frame_idx / frame_count, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Converting: {frame_idx}/{frame_count} frames")

            # Cleanup
            progress_bar.empty()
            status_text.empty()
            cap.release()
            out.release()

            logger.info(f"Successfully converted {filename} to H.264 MP4 ({new_width}x{new_height})")

        finally:
            # Cleanup temp input file
            if os.path.exists(input_path):
                os.unlink(input_path)

        return str(output_path), cache_filename

    except Exception as e:
        logger.error(f"Video conversion failed: {str(e)}")
        # Cleanup on error
        try:
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            if 'input_path' in locals() and os.path.exists(input_path):
                os.unlink(input_path)
            if output_path.exists():
                os.unlink(output_path)
        except:
            pass
        raise


def needs_conversion(filename: str) -> bool:
    """Check if file needs conversion (not already H.264 MP4)"""
    # Files that typically need conversion
    needs_conv_extensions = ['.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    return any(filename.lower().endswith(ext) for ext in needs_conv_extensions)


def get_conversion_message(filename: str) -> str:
    """Get user-friendly message about video format"""
    ext = Path(filename).suffix.lower()

    messages = {
        '.avi': "AVI format detected. Converting to MP4 for playback...",
        '.mov': "MOV format detected. Converting to MP4 for playback...",
        '.mkv': "MKV format detected. Converting to MP4 for playback...",
        '.flv': "FLV format detected. Converting to MP4 for playback...",
        '.wmv': "WMV format detected. Converting to MP4 for playback...",
        '.webm': "WebM format detected. Converting to MP4 for playback...",
        '.m4v': "M4V format detected. Converting to MP4 for playback...",
    }

    return messages.get(ext, "Converting video format for optimal playback...")