from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from src.config import settings, yaml_config, get_project_root
from src.utils import get_logger


logger = get_logger(__name__)


class DatasetLoader:
    def __init__(self):
        self.root_dir = get_project_root()
        self.data_config = yaml_config.get("data", {})
        self.datasets_config = self.data_config.get("datasets", {})
        
    def load_hatexplain(self, split: str = "train") -> Dataset:
        """
        Load hate speech dataset (UC Berkeley Measuring Hate Speech as alternative to HateXplain)
        
        Args:
            split: Dataset split to load ('train', 'validation', or 'test')
            
        Returns:
            Dataset with 'text' and 'hatespeech' fields
            - text: The text content
            - hatespeech: 0=not hate, 1=maybe hate, 2=hate speech
        """
        logger.info(f"Loading hate speech dataset (split: {split})")
        
        try:
            # Try loading from local cache first (alternative dataset)
            cache_dir = self.root_dir / "cache" / "hatexplain"
            
            if cache_dir.exists():
                logger.info(f"Loading from local cache: {cache_dir}")
                dataset = load_dataset(str(cache_dir), split=split)
                logger.info(f"Loaded {len(dataset)} samples from cached dataset")
                logger.info("Dataset uses 'hatespeech' field: 0=not hate, 1=maybe, 2=hate")
                return dataset
            else:
                # Fallback: try loading from HuggingFace (will likely fail with deprecated datasets)
                logger.warning("Local cache not found, attempting to load from HuggingFace...")
                logger.warning("This may fail if the dataset uses deprecated loading scripts")
                
                try:
                    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", split=split)
                    logger.info(f"Loaded {len(dataset)} samples from HuggingFace")
                    return dataset
                except Exception as hf_error:
                    logger.error(f"Failed to load from HuggingFace: {hf_error}")
                    logger.error("Please run: python scripts/download_datasets.py")
                    raise
                    
        except Exception as e:
            logger.error(f"Failed to load hate speech dataset: {e}")
            logger.error("Make sure to download the dataset first: python scripts/download_datasets.py")
            raise
    
    def load_deepfake_dataset(self, data_dir: Optional[str] = None) -> pd.DataFrame:
        logger.info("Loading deepfake detection dataset")
        
        if data_dir is None:
            data_dir = self.root_dir / self.data_config.get("raw_dir", "data/raw") / "deepfake"
        else:
            data_dir = Path(data_dir)
        
        if not data_dir.exists():
            logger.warning(f"Deepfake dataset directory not found: {data_dir}")
            logger.info("Please download the dataset manually from Kaggle or FaceForensics++")
            return pd.DataFrame()
        
        metadata_file = data_dir / "metadata.json"
        if metadata_file.exists():
            df = pd.read_json(metadata_file)
            logger.info(f"Loaded {len(df)} samples from deepfake dataset")
            return df
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
            return pd.DataFrame()
    
    def load_custom_text_data(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Loading custom text data from {file_path}")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            df = pd.read_json(file_path)
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded {len(df)} samples from {file_path}")
        return df
    
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        batch_size = batch_size or settings.BATCH_SIZE
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        return self.datasets_config.get(dataset_name, {})