import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_from_disk
from tqdm import tqdm

from src.models import HateSpeechDetector
from src.data import TextPreprocessor
from src.config import yaml_config, get_project_root, settings
from src.utils import get_logger


logger = get_logger(__name__)


def train_nlp_model():
    logger.info("Starting NLP model training...")
    
    nlp_config = yaml_config["models"]["nlp"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = HateSpeechDetector(
        model_name=nlp_config["base_model"],
        num_labels=nlp_config["num_labels"]
    ).to(device)
    
    preprocessor = TextPreprocessor(nlp_config["base_model"])
    
    cache_dir = get_project_root() / settings.CACHE_DIR / "hatexplain"
    
    try:
        train_dataset = load_from_disk(str(cache_dir / "train"))
        val_dataset = load_from_disk(str(cache_dir / "validation"))
        logger.info(f"Loaded datasets from cache")
    except:
        logger.info("Loading datasets from HuggingFace...")
        from datasets import load_dataset
        train_dataset = load_dataset("hatexplain", split="train")
        val_dataset = load_dataset("hatexplain", split="validation")
    
    def collate_fn(batch):
        texts = [item["post_tokens"] if isinstance(item["post_tokens"], str) else " ".join(item["post_tokens"]) for item in batch]
        labels = [item["annotators"]["label"][0] if isinstance(item["annotators"]["label"], list) else item["annotators"]["label"] for item in batch]
        
        encodings = preprocessor.tokenize(texts)
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=nlp_config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=nlp_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=nlp_config["learning_rate"],
        weight_decay=nlp_config["weight_decay"]
    )
    
    total_steps = len(train_loader) * nlp_config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=nlp_config["warmup_steps"],
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(nlp_config["epochs"]):
        logger.info(f"Epoch {epoch + 1}/{nlp_config['epochs']}")
        
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"]
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs["loss"].item()
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Average validation loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            models_dir = get_project_root() / settings.MODELS_DIR
            models_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = models_dir / "hate_speech_detector.pth"
            model.save_model(str(save_path))
            
            logger.info(f"Model saved to {save_path}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    train_nlp_model()