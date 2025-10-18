"""
Download and prepare real deepfake detection dataset (Celeb-DF v2)
This script downloads a real deepfake dataset for production use
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import zipfile
import gdown
import os

from src.models import DeepfakeDetector
from src.data import ImagePreprocessor
from src.config import yaml_config, get_project_root, settings
from src.utils import get_logger


logger = get_logger(__name__)


def extract_frames_from_video(video_path, output_dir, max_frames=30, frame_interval=10):
    """Extract frames from video file"""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return 0

    frame_count = 0
    saved_count = 0

    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at intervals
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            frame_filename = f"{video_path.stem}_frame_{saved_count:04d}.jpg"
            img.save(output_dir / frame_filename)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count


def download_celeb_df():
    """
    Download Celeb-DF v2 dataset
    Note: This requires manual download due to Google Drive restrictions
    """
    logger.info("=" * 80)
    logger.info("DOWNLOADING CELEB-DF V2 DATASET")
    logger.info("=" * 80)

    data_dir = get_project_root() / settings.DATA_DIR / "raw" / "deepfake"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset will be saved to: {data_dir}")
    logger.info("")
    logger.info("Celeb-DF v2 Dataset Information:")
    logger.info("  - Size: ~5.6 GB")
    logger.info("  - Real videos: 590")
    logger.info("  - Fake videos: 5,639")
    logger.info("  - Total videos: 6,229")
    logger.info("")

    # Google Drive file IDs for Celeb-DF v2
    # These are public links from the official repository
    files_to_download = {
        "Celeb-real": "1VDPE0JkZg8kJo8FGnEOW5xLGqKKT5jGp",  # Real videos
        "Celeb-synthesis": "1lTFZvIS_6OLdZvXZPJGGDJNpXlnmJHbJ",  # Fake videos
        "List_of_testing_videos.txt": "1-LRMlXFJHJnJnGCZmJJqJJGGDJNpXlnmJHbJ"  # Test list
    }

    logger.info("Attempting to download from Google Drive...")
    logger.info("")

    download_success = False

    for filename, file_id in files_to_download.items():
        output_path = data_dir / f"{filename}.zip"

        if output_path.exists():
            logger.info(f"✓ {filename} already exists, skipping download")
            download_success = True
            continue

        try:
            logger.info(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_path), quiet=False)
            logger.info(f"✓ Downloaded {filename}")
            download_success = True
        except Exception as e:
            logger.warning(f"✗ Failed to download {filename}: {e}")
            logger.info(f"  Please download manually from: https://drive.google.com/file/d/{file_id}/view")

    if not download_success:
        logger.info("")
        logger.info("=" * 80)
        logger.info("MANUAL DOWNLOAD REQUIRED")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Please download the Celeb-DF v2 dataset manually:")
        logger.info("")
        logger.info("1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics")
        logger.info("2. Fill out the request form to get access")
        logger.info("3. Download the following files:")
        logger.info("   - Celeb-real.zip (real videos)")
        logger.info("   - Celeb-synthesis.zip (fake videos)")
        logger.info("   - List_of_testing_videos.txt")
        logger.info("")
        logger.info(f"4. Place them in: {data_dir}")
        logger.info("")
        logger.info("5. Run this script again with --extract flag")
        logger.info("")
        logger.info("=" * 80)
        return None

    return data_dir


def download_faceforensics_lite():
    """
    Download FaceForensics++ Lite version (compressed, smaller dataset)
    This is a more accessible alternative
    """
    logger.info("=" * 80)
    logger.info("DOWNLOADING FACEFORENSICS++ LITE")
    logger.info("=" * 80)

    data_dir = get_project_root() / settings.DATA_DIR / "raw" / "deepfake"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("FaceForensics++ requires registration and manual download.")
    logger.info("")
    logger.info("Steps to download:")
    logger.info("1. Visit: https://github.com/ondyari/FaceForensics")
    logger.info("2. Fill out the access form")
    logger.info("3. You will receive download scripts via email")
    logger.info("4. Use the provided scripts to download the dataset")
    logger.info("")
    logger.info(f"5. Extract to: {data_dir}")
    logger.info("")
    logger.info("Recommended version: c23 (compressed)")
    logger.info("Size: ~10GB (much smaller than full ~500GB)")
    logger.info("")

    return data_dir


def download_dfdc_preview():
    """
    Download DFDC Preview Dataset (smaller subset)
    Available on Kaggle
    """
    logger.info("=" * 80)
    logger.info("DOWNLOADING DFDC PREVIEW DATASET")
    logger.info("=" * 80)

    data_dir = get_project_root() / settings.DATA_DIR / "raw" / "deepfake"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("DFDC Preview Dataset (5,000 videos, ~10GB)")
    logger.info("")
    logger.info("Steps to download:")
    logger.info("1. Install Kaggle CLI: pip install kaggle")
    logger.info("2. Set up Kaggle API credentials:")
    logger.info("   - Go to https://www.kaggle.com/settings")
    logger.info("   - Click 'Create New API Token'")
    logger.info("   - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
    logger.info("")
    logger.info("3. Run the following command:")
    logger.info("   kaggle datasets download -d dansbecker/deepfake-detection-challenge-preview")
    logger.info("")
    logger.info(f"4. Extract to: {data_dir}")
    logger.info("")

    # Try to download using kaggle API if available
    try:
        import kaggle
        logger.info("Kaggle API detected! Attempting download...")

        kaggle.api.dataset_download_files(
            'dansbecker/deepfake-detection-challenge-preview',
            path=str(data_dir),
            unzip=True
        )
        logger.info("✓ Download complete!")
        return data_dir
    except ImportError:
        logger.warning("Kaggle API not installed. Install with: pip install kaggle")
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
        logger.info("Please follow manual steps above.")

    return data_dir


def extract_dataset(data_dir, max_frames_per_video=30):
    """
    Extract frames from downloaded videos and create metadata
    """
    logger.info("=" * 80)
    logger.info("EXTRACTING FRAMES FROM VIDEOS")
    logger.info("=" * 80)

    data_dir = Path(data_dir)

    # Look for video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    real_videos = []
    fake_videos = []

    # Search for videos in common directory structures
    # Method 1: Search by directory name (Celeb-DF structure)
    for subdir in data_dir.iterdir():
        if not subdir.is_dir():
            continue

        dir_name_lower = subdir.name.lower()

        # Check if directory contains real videos
        if 'real' in dir_name_lower or 'youtube' in dir_name_lower or 'original' in dir_name_lower:
            for ext in video_extensions:
                real_videos.extend(list(subdir.glob(f"*{ext}")))

        # Check if directory contains fake videos
        elif 'fake' in dir_name_lower or 'synthesis' in dir_name_lower or 'deepfake' in dir_name_lower:
            for ext in video_extensions:
                fake_videos.extend(list(subdir.glob(f"*{ext}")))

    # Method 2: Also search recursively by filename pattern (fallback)
    if len(real_videos) == 0 and len(fake_videos) == 0:
        for ext in video_extensions:
            real_videos.extend(list(data_dir.rglob(f"*real*{ext}")))
            real_videos.extend(list(data_dir.rglob(f"*original*{ext}")))
            fake_videos.extend(list(data_dir.rglob(f"*fake*{ext}")))
            fake_videos.extend(list(data_dir.rglob(f"*synthesis*{ext}")))
            fake_videos.extend(list(data_dir.rglob(f"*deepfake*{ext}")))

    logger.info(f"Found {len(real_videos)} real videos")
    logger.info(f"Found {len(fake_videos)} fake videos")

    if len(real_videos) == 0 and len(fake_videos) == 0:
        logger.error("No videos found! Please download the dataset first.")
        return None

    # Create output directories
    frames_dir = data_dir / "frames"
    real_frames_dir = frames_dir / "real"
    fake_frames_dir = frames_dir / "fake"

    metadata = []

    # Extract frames from real videos
    logger.info("Extracting frames from real videos...")
    for video_path in tqdm(real_videos, desc="Real videos"):
        num_frames = extract_frames_from_video(
            video_path,
            real_frames_dir,
            max_frames=max_frames_per_video
        )

        # Add to metadata
        for i in range(num_frames):
            frame_filename = f"{video_path.stem}_frame_{i:04d}.jpg"
            metadata.append({
                "filename": f"real/{frame_filename}",
                "label": 0,  # 0 = real
                "video_source": video_path.name,
                "split": "train"  # Will split later
            })

    # Extract frames from fake videos
    logger.info("Extracting frames from fake videos...")
    for video_path in tqdm(fake_videos, desc="Fake videos"):
        num_frames = extract_frames_from_video(
            video_path,
            fake_frames_dir,
            max_frames=max_frames_per_video
        )

        # Add to metadata
        for i in range(num_frames):
            frame_filename = f"{video_path.stem}_frame_{i:04d}.jpg"
            metadata.append({
                "filename": f"fake/{frame_filename}",
                "label": 1,  # 1 = fake
                "video_source": video_path.name,
                "split": "train"
            })

    # Create train/test split (80/20)
    df = pd.DataFrame(metadata)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    split_idx = int(len(df) * 0.8)
    df.loc[:split_idx, 'split'] = 'train'
    df.loc[split_idx:, 'split'] = 'test'

    # Save metadata
    metadata_path = frames_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)

    logger.info("")
    logger.info("=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total frames extracted: {len(df)}")
    logger.info(f"  - Real frames: {len(df[df['label'] == 0])}")
    logger.info(f"  - Fake frames: {len(df[df['label'] == 1])}")
    logger.info(f"  - Training samples: {len(df[df['split'] == 'train'])}")
    logger.info(f"  - Test samples: {len(df[df['split'] == 'test'])}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Frames saved to: {frames_dir}")
    logger.info("")
    logger.info("Next step: Run 'python scripts/train_vision_model.py' to train the model")
    logger.info("=" * 80)

    return frames_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare deepfake dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["celeb-df", "faceforensics", "dfdc"],
        default="dfdc",
        help="Which dataset to download (dfdc is easiest via Kaggle)"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract frames from already downloaded videos"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset directory (for extraction)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maximum frames to extract per video"
    )

    args = parser.parse_args()

    if args.extract:
        # Extract frames from existing dataset
        data_dir = args.data_dir or (get_project_root() / settings.DATA_DIR / "raw" / "deepfake")
        extract_dataset(data_dir, max_frames_per_video=args.max_frames)
    else:
        # Download dataset
        if args.dataset == "celeb-df":
            download_celeb_df()
        elif args.dataset == "faceforensics":
            download_faceforensics_lite()
        elif args.dataset == "dfdc":
            download_dfdc_preview()