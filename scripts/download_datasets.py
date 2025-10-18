import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from src.config import get_project_root, settings
from src.utils import get_logger


logger = get_logger(__name__)


def download_hatexplain():
    logger.info("Downloading HateXplain dataset...")
    
    # The original HateXplain dataset is no longer available in the old format
    # We'll use the alternative UC Berkeley dataset which is more reliable
    logger.info("Using UC Berkeley Measuring Hate Speech dataset (more reliable and larger)")
    
    try:
        download_alternative_dataset()
    except Exception as e:
        logger.error(f"Failed to download hate speech dataset: {e}")
        raise RuntimeError("Failed to download hate speech dataset")


def download_alternative_dataset():
    """Download alternative hate speech dataset"""
    logger.info("Downloading UC Berkeley Measuring Hate Speech dataset...")
    
    try:
        from datasets import DatasetDict
        
        # Load the measuring-hate-speech dataset (Parquet format, reliable)
        logger.info("Loading dataset from HuggingFace...")
        dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
        
        # Split into train/val/test (80/10/10)
        logger.info("Splitting dataset into train/validation/test...")
        train_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
        val_test = train_test["test"].train_test_split(test_size=0.5, seed=42)
        
        train_dataset = train_test["train"]
        val_dataset = val_test["train"]
        test_dataset = val_test["test"]
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Map the dataset to our expected format
        # UC Berkeley dataset uses hate_speech_score ranging from -7.61 to +4.01
        # Negative scores = respectful/normal, Positive scores = offensive/hateful
        # We'll convert it to 3 classes: 0=normal, 1=offensive, 2=hate
        def map_to_hatexplain_format(example):
            # Convert hate_speech_score to our 3-class format
            # Based on actual distribution: mean=-1.03, median=-0.73, range=[-7.61, 4.01]
            score = example.get("hate_speech_score", -5.0)
            
            # Robust thresholds based on score distribution:
            # < -1.0: Clearly respectful/normal content
            # -1.0 to 1.0: Borderline/offensive (insults, profanity, disrespect)
            # > 1.0: Hate speech (dehumanization, violence, targeted attacks)
            if score < -1.0:
                label = 0  # normal (respectful, neutral)
            elif score < 1.0:
                label = 1  # offensive (insults, disrespect, profanity)
            else:
                label = 2  # hate (dehumanization, violence, targeted hate)
            
            return {
                "post_tokens": example["text"],
                "annotators": {"label": label}
            }
        
        logger.info("Converting dataset to HateXplain format...")
        train_dataset = train_dataset.map(map_to_hatexplain_format)
        val_dataset = val_dataset.map(map_to_hatexplain_format)
        test_dataset = test_dataset.map(map_to_hatexplain_format)
        
        cache_dir = get_project_root() / settings.CACHE_DIR / "hatexplain"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each split separately (matching the training script's expectations)
        # Use absolute paths with proper Windows formatting
        logger.info("Saving datasets to disk...")
        train_path = str(cache_dir / "train").replace("/", "\\")
        val_path = str(cache_dir / "validation").replace("/", "\\")
        test_path = str(cache_dir / "test").replace("/", "\\")
        
        train_dataset.save_to_disk(train_path)
        val_dataset.save_to_disk(val_path)
        test_dataset.save_to_disk(test_path)
        
        logger.info(f"[SUCCESS] Dataset saved to {cache_dir}")
        logger.info("Dataset format: post_tokens (text), annotators.label (0=normal, 1=offensive, 2=hate)")
        logger.info(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
        
    except Exception as e:
        logger.error(f"Failed to download alternative dataset: {e}")
        raise


def download_deepfake_info():
    logger.info("Deepfake dataset information:")
    logger.info("Please download manually from:")
    logger.info("1. FaceForensics++: https://github.com/ondyari/FaceForensics")
    logger.info("2. DFDC: https://www.kaggle.com/c/deepfake-detection-challenge")
    logger.info(f"Save to: {get_project_root() / settings.DATA_DIR / 'raw' / 'deepfake'}")


def main():
    logger.info("Starting dataset download...")
    
    download_hatexplain()
    download_deepfake_info()
    
    logger.info("Dataset download complete!")


if __name__ == "__main__":
    main()