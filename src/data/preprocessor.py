import re
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import AutoTokenizer
from torchvision import transforms
from langdetect import detect, LangDetectException

from src.config import settings, yaml_config
from src.utils import get_logger


logger = get_logger(__name__)


class TextPreprocessor:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.NLP_MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_length = settings.MAX_LENGTH
        
    def clean_text(self, text: str) -> str:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', lambda m: m.group(0)[1:], text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            logger.warning("Language detection failed, defaulting to 'en'")
            return "en"
    
    def tokenize(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        encodings = self.tokenizer(
            cleaned_texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
        
        return encodings
    
    def preprocess_batch(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenize(batch)
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


class ImagePreprocessor:
    def __init__(self, image_size: int = 224, augmentation: bool = False):
        self.image_size = image_size
        self.augmentation = augmentation
        
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        if augmentation:
            self.train_transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.train_transform = self.base_transform
    
    def load_image(self, image_path: str) -> Image.Image:
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image, training: bool = False) -> torch.Tensor:
        transform = self.train_transform if training else self.base_transform
        return transform(image)
    
    def preprocess_video_frame(self, frame: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return self.preprocess_image(pil_image)
    
    def extract_video_frames(
        self,
        video_path: str,
        num_frames: int = 10,
        method: str = "uniform"
    ) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if method == "uniform":
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif method == "random":
            frame_indices = np.random.choice(total_frames, num_frames, replace=False)
            frame_indices.sort()
        else:
            raise ValueError(f"Unknown frame extraction method: {method}")
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def preprocess_video(
        self,
        video_path: str,
        num_frames: int = 10
    ) -> torch.Tensor:
        frames = self.extract_video_frames(video_path, num_frames)
        processed_frames = [self.preprocess_video_frame(frame) for frame in frames]
        return torch.stack(processed_frames)