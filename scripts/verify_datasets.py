"""
Verify that all datasets are properly downloaded and ready for training.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_project_root, settings
from src.utils import get_logger

logger = get_logger(__name__)


def check_hate_speech_dataset() -> Tuple[bool, Dict]:
    """Check if hate speech dataset is available"""
    cache_dir = get_project_root() / settings.CACHE_DIR / "hatexplain"
    
    status = {
        "available": False,
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
        "total_samples": 0,
        "location": str(cache_dir)
    }
    
    try:
        from datasets import load_from_disk
        
        # Check if dataset exists
        if not cache_dir.exists():
            logger.warning(f"❌ Hate speech dataset not found at: {cache_dir}")
            return False, status
        
        # Try to load dataset
        train_path = cache_dir / "train"
        val_path = cache_dir / "validation"
        test_path = cache_dir / "test"
        
        if train_path.exists() and val_path.exists() and test_path.exists():
            train_dataset = load_from_disk(str(train_path))
            val_dataset = load_from_disk(str(val_path))
            test_dataset = load_from_disk(str(test_path))
            
            status["available"] = True
            status["train_samples"] = len(train_dataset)
            status["val_samples"] = len(val_dataset)
            status["test_samples"] = len(test_dataset)
            status["total_samples"] = len(train_dataset) + len(val_dataset) + len(test_dataset)
            
            logger.info(f"✅ Hate Speech Dataset: {status['total_samples']:,} samples")
            logger.info(f"   Train: {status['train_samples']:,}")
            logger.info(f"   Validation: {status['val_samples']:,}")
            logger.info(f"   Test: {status['test_samples']:,}")
            
            return True, status
        else:
            logger.warning("❌ Hate speech dataset incomplete (missing splits)")
            return False, status
            
    except Exception as e:
        logger.error(f"❌ Error loading hate speech dataset: {e}")
        return False, status


def check_deepfake_dataset() -> Tuple[bool, Dict]:
    """Check if deepfake dataset is available"""
    data_dir = get_project_root() / settings.DATA_DIR / "raw" / "deepfake"
    
    status = {
        "available": False,
        "videos_count": 0,
        "frames_count": 0,
        "real_count": 0,
        "fake_count": 0,
        "location": str(data_dir)
    }
    
    try:
        # Check for videos
        videos_dir = data_dir / "videos"
        frames_dir = data_dir / "frames"
        sample_dir = data_dir / "sample"
        
        # Check multiple possible locations
        found_data = False
        
        if frames_dir.exists():
            # Count frames
            real_frames = list(frames_dir.glob("**/real/*.jpg")) + list(frames_dir.glob("**/real/*.png"))
            fake_frames = list(frames_dir.glob("**/fake/*.jpg")) + list(frames_dir.glob("**/fake/*.png"))
            
            status["frames_count"] = len(real_frames) + len(fake_frames)
            status["real_count"] = len(real_frames)
            status["fake_count"] = len(fake_frames)
            
            if status["frames_count"] > 0:
                found_data = True
                logger.info(f"✅ Deepfake Frames: {status['frames_count']:,} frames")
                logger.info(f"   Real: {status['real_count']:,}")
                logger.info(f"   Fake: {status['fake_count']:,}")
        
        if videos_dir.exists():
            # Count videos
            videos = list(videos_dir.glob("**/*.mp4")) + list(videos_dir.glob("**/*.avi"))
            status["videos_count"] = len(videos)
            
            if status["videos_count"] > 0:
                found_data = True
                logger.info(f"✅ Deepfake Videos: {status['videos_count']:,} videos")
        
        if sample_dir.exists():
            # Check sample dataset
            sample_files = list(sample_dir.glob("**/*.*"))
            if len(sample_files) > 0:
                found_data = True
                logger.info(f"✅ Deepfake Sample: {len(sample_files):,} files")
        
        if not found_data:
            logger.warning("❌ Deepfake dataset not found")
            logger.info("   Run: python scripts/download_deepfake_sample.py")
            logger.info("   Or see: DATASET_GUIDE.md for FaceForensics++ download")
            return False, status
        
        status["available"] = found_data
        return found_data, status
        
    except Exception as e:
        logger.error(f"❌ Error checking deepfake dataset: {e}")
        return False, status


def check_gdelt_access() -> Tuple[bool, Dict]:
    """Check if GDELT API is accessible"""
    status = {
        "available": False,
        "api_accessible": False,
        "sample_events": 0
    }
    
    try:
        import requests
        
        # Test GDELT API
        test_url = "https://api.gdeltproject.org/api/v2/doc/doc?query=test&mode=artlist&maxrecords=1&format=json"
        
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            status["api_accessible"] = True
            status["available"] = True
            logger.info("✅ GDELT API: Connected")
        else:
            logger.warning(f"❌ GDELT API returned status: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        logger.warning(f"❌ GDELT API not accessible: {e}")
        logger.info("   This is optional - system can work without GDELT")
    except Exception as e:
        logger.error(f"❌ Error testing GDELT: {e}")
    
    # Check for cached GDELT data
    gdelt_dir = get_project_root() / settings.DATA_DIR / "raw" / "gdelt"
    if gdelt_dir.exists():
        events = list(gdelt_dir.glob("**/*.json"))
        status["sample_events"] = len(events)
        if len(events) > 0:
            logger.info(f"✅ GDELT Cache: {len(events):,} cached events")
    
    return status["available"], status


def check_models() -> Tuple[bool, Dict]:
    """Check if trained models are available"""
    models_dir = get_project_root() / settings.MODELS_DIR
    
    status = {
        "hate_speech_model": False,
        "deepfake_model": False,
        "fusion_model": False,
        "location": str(models_dir)
    }
    
    if not models_dir.exists():
        logger.warning("❌ Models directory not found")
        logger.info("   Models will be created during training")
        return False, status
    
    # Check for model files
    hate_speech_model = models_dir / "hate_speech_detector.pt"
    deepfake_model = models_dir / "deepfake_detector.pt"
    fusion_model = models_dir / "fusion_model.pt"
    
    if hate_speech_model.exists():
        status["hate_speech_model"] = True
        logger.info(f"✅ Hate Speech Model: {hate_speech_model.stat().st_size / 1024 / 1024:.1f} MB")
    
    if deepfake_model.exists():
        status["deepfake_model"] = True
        logger.info(f"✅ Deepfake Model: {deepfake_model.stat().st_size / 1024 / 1024:.1f} MB")
    
    if fusion_model.exists():
        status["fusion_model"] = True
        logger.info(f"✅ Fusion Model: {fusion_model.stat().st_size / 1024 / 1024:.1f} MB")
    
    if not any([status["hate_speech_model"], status["deepfake_model"], status["fusion_model"]]):
        logger.warning("❌ No trained models found")
        logger.info("   Run training scripts to create models:")
        logger.info("   - python scripts/train_nlp_model.py")
        logger.info("   - python scripts/train_vision_model.py")
        return False, status
    
    return True, status


def print_summary(results: Dict):
    """Print summary of dataset verification"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    total_checks = len(results)
    passed_checks = sum(1 for r in results.values() if r[0])
    
    logger.info(f"Checks Passed: {passed_checks}/{total_checks}")
    logger.info("")
    
    # Hate Speech
    hate_speech_ok, hate_speech_status = results["hate_speech"]
    if hate_speech_ok:
        logger.info(f"✅ Hate Speech Dataset: {hate_speech_status['total_samples']:,} samples")
    else:
        logger.info("❌ Hate Speech Dataset: NOT FOUND")
        logger.info("   → Run: python scripts/download_datasets.py")
    
    # Deepfake
    deepfake_ok, deepfake_status = results["deepfake"]
    if deepfake_ok:
        if deepfake_status["frames_count"] > 0:
            logger.info(f"✅ Deepfake Dataset: {deepfake_status['frames_count']:,} frames")
        elif deepfake_status["videos_count"] > 0:
            logger.info(f"✅ Deepfake Dataset: {deepfake_status['videos_count']:,} videos")
    else:
        logger.info("❌ Deepfake Dataset: NOT FOUND")
        logger.info("   → Run: python scripts/download_deepfake_sample.py")
        logger.info("   → Or see: DATASET_GUIDE.md")
    
    # GDELT
    gdelt_ok, gdelt_status = results["gdelt"]
    if gdelt_ok:
        logger.info("✅ GDELT API: Connected")
    else:
        logger.info("⚠️  GDELT API: Not accessible (optional)")
    
    # Models
    models_ok, models_status = results["models"]
    if models_ok:
        logger.info("✅ Trained Models: Available")
    else:
        logger.info("⚠️  Trained Models: Not found (will be created during training)")
    
    logger.info("")
    logger.info("=" * 60)
    
    if passed_checks >= 2:  # At least hate speech + one other
        logger.info("✅ READY FOR TRAINING!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Train models: python scripts/train_nlp_model.py")
        logger.info("  2. Run dashboard: streamlit run src/dashboard/app.py")
        logger.info("  3. See DATASET_GUIDE.md for full reproduction")
    elif hate_speech_ok:
        logger.info("⚠️  PARTIAL SETUP")
        logger.info("")
        logger.info("You can train hate speech detection, but need more data for:")
        logger.info("  - Deepfake detection (download sample or FaceForensics++)")
        logger.info("  - Event correlation (GDELT API access)")
        logger.info("")
        logger.info("See DATASET_GUIDE.md for instructions")
    else:
        logger.info("❌ SETUP INCOMPLETE")
        logger.info("")
        logger.info("Required: Download hate speech dataset")
        logger.info("  → Run: python scripts/download_datasets.py")
        logger.info("")
        logger.info("See DATASET_GUIDE.md for complete setup instructions")
    
    logger.info("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify datasets are ready")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    
    args = parser.parse_args()
    
    logger.info("Verifying datasets...")
    logger.info("")
    
    results = {
        "hate_speech": check_hate_speech_dataset(),
        "deepfake": check_deepfake_dataset(),
        "gdelt": check_gdelt_access(),
        "models": check_models()
    }
    
    print_summary(results)
    
    # Exit code: 0 if at least hate speech is ready, 1 otherwise
    hate_speech_ok = results["hate_speech"][0]
    sys.exit(0 if hate_speech_ok else 1)


if __name__ == "__main__":
    main()