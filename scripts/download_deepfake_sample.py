"""
Download a sample deepfake dataset for testing without full FaceForensics++ download.
Uses Celeb-DF dataset which is smaller and publicly available.
"""

import sys
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_project_root, settings
from src.utils import get_logger

logger = get_logger(__name__)


def download_file(url: str, destination: Path, description: str = "Downloading"):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def download_celeb_df_sample():
    """
    Download Celeb-DF sample dataset (smaller alternative to FaceForensics++)
    
    Note: This is a sample dataset for testing. For full paper reproduction,
    you need FaceForensics++. See DATASET_GUIDE.md for instructions.
    """
    logger.info("Downloading Celeb-DF sample dataset...")
    logger.info("This is a smaller alternative for testing purposes.")
    logger.info("For full paper reproduction, download FaceForensics++ manually.")
    
    data_dir = get_project_root() / settings.DATA_DIR / "raw" / "deepfake"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample structure
    sample_dir = data_dir / "sample"
    sample_dir.mkdir(exist_ok=True)
    
    real_dir = sample_dir / "real"
    fake_dir = sample_dir / "fake"
    real_dir.mkdir(exist_ok=True)
    fake_dir.mkdir(exist_ok=True)
    
    logger.info(f"Sample dataset will be saved to: {sample_dir}")
    
    # For now, create placeholder structure
    # In production, you would download from a public source
    logger.warning("⚠️  MANUAL DOWNLOAD REQUIRED")
    logger.info("")
    logger.info("To get a sample deepfake dataset for testing:")
    logger.info("")
    logger.info("Option 1: Celeb-DF (Recommended for testing)")
    logger.info("  1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics")
    logger.info("  2. Download the dataset (requires form submission)")
    logger.info("  3. Extract to: " + str(sample_dir))
    logger.info("")
    logger.info("Option 2: DFDC Preview Dataset (Kaggle)")
    logger.info("  1. Visit: https://www.kaggle.com/c/deepfake-detection-challenge")
    logger.info("  2. Download preview dataset (~5GB)")
    logger.info("  3. Extract to: " + str(sample_dir))
    logger.info("")
    logger.info("Option 3: Use our pre-extracted sample frames")
    logger.info("  1. Download from: [Your release URL]")
    logger.info("  2. Extract to: " + str(sample_dir))
    logger.info("")
    logger.info("For FULL paper reproduction:")
    logger.info("  - Download FaceForensics++ (see DATASET_GUIDE.md)")
    logger.info("  - Register at: https://github.com/ondyari/FaceForensics")
    logger.info("")
    
    # Create README in the directory
    readme_content = """# Deepfake Sample Dataset

## Quick Start

This directory should contain sample deepfake videos/frames for testing.

### Recommended Structure:

```
sample/
├── real/           # Real (authentic) videos/frames
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── fake/           # Fake (manipulated) videos/frames
    ├── video_001.mp4
    ├── video_002.mp4
    └── ...
```

### Download Options:

1. **Celeb-DF** (Recommended for testing)
   - URL: https://github.com/yuezunli/celeb-deepfakeforensics
   - Size: ~5 GB
   - Videos: 590 real + 5,639 fake

2. **DFDC Preview** (Kaggle)
   - URL: https://www.kaggle.com/c/deepfake-detection-challenge
   - Size: ~5 GB
   - Videos: 400 videos (mixed)

3. **FaceForensics++** (Full paper reproduction)
   - URL: https://github.com/ondyari/FaceForensics
   - Size: ~150 GB (c23 compression)
   - Videos: 1,000 real + 4,000 fake

### After Download:

1. Extract videos to `real/` and `fake/` directories
2. Run frame extraction:
   ```
   python scripts/extract_deepfake_frames.py
   ```
3. Verify dataset:
   ```
   python scripts/verify_datasets.py
   ```

### For Testing Without Download:

The system can run with just the hate speech dataset.
Deepfake detection will be disabled until you add video data.
"""
    
    readme_path = sample_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"✅ Created sample directory structure at: {sample_dir}")
    logger.info(f"✅ Created README with download instructions")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Follow instructions in: " + str(readme_path))
    logger.info("  2. Download one of the suggested datasets")
    logger.info("  3. Run: python scripts/verify_datasets.py")


def create_mock_dataset():
    """
    Create a tiny mock dataset for CI/CD testing
    (Not suitable for actual training)
    """
    logger.info("Creating mock dataset for testing...")
    
    data_dir = get_project_root() / settings.DATA_DIR / "raw" / "deepfake" / "mock"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    real_dir.mkdir(exist_ok=True)
    fake_dir.mkdir(exist_ok=True)
    
    # Create placeholder files
    for i in range(10):
        (real_dir / f"real_{i:03d}.txt").write_text(f"Mock real image {i}")
        (fake_dir / f"fake_{i:03d}.txt").write_text(f"Mock fake image {i}")
    
    logger.info(f"✅ Created mock dataset at: {data_dir}")
    logger.info("⚠️  This is only for testing - not suitable for training!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download sample deepfake dataset")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Create mock dataset for testing (not for training)"
    )
    
    args = parser.parse_args()
    
    if args.mock:
        create_mock_dataset()
    else:
        download_celeb_df_sample()


if __name__ == "__main__":
    main()