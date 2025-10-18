"""
Train DeepFake Detection Model (Vision Model)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
from PIL import Image

from src.models import DeepfakeDetector
from src.data import ImagePreprocessor
from src.config import yaml_config, get_project_root, settings
from src.utils import get_logger


logger = get_logger(__name__)


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection"""

    def __init__(self, data_dir, metadata_df, preprocessor, training=True):
        self.data_dir = Path(data_dir)
        self.metadata = metadata_df
        self.preprocessor = preprocessor
        self.training = training

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Handle different path structures
        if 'filename' in row:
            image_path = self.data_dir / row["filename"]
        else:
            # Fallback for different metadata formats
            image_path = self.data_dir / row.get("path", row.get("file", ""))

        try:
            image = self.preprocessor.load_image(str(image_path))
            image_tensor = self.preprocessor.preprocess_image(image, training=self.training)
            label = int(row["label"])

            return image_tensor, label
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return a dummy tensor and label
            dummy_tensor = torch.zeros(3, self.preprocessor.image_size, self.preprocessor.image_size)
            return dummy_tensor, 0


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # Handle dict output from model
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Handle dict output from model
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            loss = criterion(logits, labels)

            total_loss += loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_vision_model(
    data_dir=None,
    epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    save_path=None
):
    """Train the deepfake detection model"""

    logger.info("=" * 80)
    logger.info("TRAINING DEEPFAKE DETECTION MODEL")
    logger.info("=" * 80)

    # Configuration
    vision_config = yaml_config["models"]["vision"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Initialize model
    logger.info(f"Initializing model: {vision_config['base_model']}")
    model = DeepfakeDetector(
        model_name=vision_config["base_model"],
        num_classes=vision_config["num_classes"]
    ).to(device)

    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        image_size=vision_config["image_size"],
        augmentation=vision_config["augmentation"]
    )

    # Find dataset
    if data_dir is None:
        # Try multiple possible locations
        possible_dirs = [
            get_project_root() / settings.DATA_DIR / "raw" / "frames",  # Celeb-DF extracted frames
            get_project_root() / settings.DATA_DIR / "raw" / "deepfake" / "frames",
            get_project_root() / settings.DATA_DIR / "raw" / "deepfake",
            get_project_root() / settings.DATA_DIR / "processed" / "deepfake",
        ]

        data_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                metadata_path = dir_path / "metadata.csv"
                if metadata_path.exists():
                    data_dir = dir_path
                    logger.info(f"Found dataset at: {data_dir}")
                    break

        if data_dir is None:
            logger.error("Dataset not found!")
            logger.error("Please run one of the following:")
            logger.error("  1. python scripts/download_deepfake_dataset.py --dataset dfdc")
            logger.error("  2. python scripts/download_deepfake_dataset.py --extract")
            logger.error("")
            logger.error("Or manually place your dataset in: data/raw/deepfake/")
            return
    else:
        data_dir = Path(data_dir)

    # Load metadata
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        logger.error("Please ensure metadata.csv exists in the dataset directory")
        return

    logger.info(f"Loading metadata from: {metadata_path}")
    metadata = pd.read_csv(metadata_path)

    logger.info(f"Total samples: {len(metadata)}")
    logger.info(f"  - Real: {len(metadata[metadata['label'] == 0])}")
    logger.info(f"  - Fake: {len(metadata[metadata['label'] == 1])}")

    # Split into train/test
    if 'split' in metadata.columns:
        train_df = metadata[metadata['split'] == 'train'].reset_index(drop=True)
        test_df = metadata[metadata['split'] == 'test'].reset_index(drop=True)
    else:
        # Create split if not exists
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            metadata,
            test_size=0.2,
            random_state=42,
            stratify=metadata['label']
        )

    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    # Create datasets
    train_dataset = DeepfakeDataset(data_dir, train_df, preprocessor, training=True)
    test_dataset = DeepfakeDataset(data_dir, test_df, preprocessor, training=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Training loop
    logger.info("")
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Optimizer: AdamW")
    logger.info("")

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        logger.info("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"✓ New best model! Validation Accuracy: {val_acc:.2f}%")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model
    if save_path is None:
        save_path = get_project_root() / settings.MODELS_DIR / "deepfake_detector.pth"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Saving model to: {save_path}")

    model.save_model(str(save_path))

    # Get file size
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model saved successfully! Size: {file_size_mb:.2f} MB")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start the API: python run_api.py")
    logger.info("  2. Test the model with your test cases")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the trained model"
    )

    args = parser.parse_args()

    train_vision_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )