import torch
import torch.nn as nn
import timm
from typing import Dict, Optional, Tuple

from src.config import yaml_config
from src.utils import get_logger


logger = get_logger(__name__)


class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        logger.info(f"Initialized DeepfakeDetector with {model_name}")

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "features": features
        }

    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        with torch.no_grad():
            outputs = self.forward(images)

        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        return predictions, probabilities

    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()

        with torch.no_grad():
            outputs = self.forward(images)

        return outputs["features"]

    def save_model(self, save_path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        checkpoint = torch.load(load_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {load_path}")


class VideoDeepfakeDetector(nn.Module):
    def __init__(
        self,
        frame_model_name: str = "efficientnet_b0",
        num_classes: int = 2,
        hidden_dim: int = 256,
        num_frames: int = 10
    ):
        super().__init__()

        self.frame_detector = DeepfakeDetector(
            model_name=frame_model_name,
            num_classes=0,
            pretrained=True
        )

        num_features = self.frame_detector.backbone.num_features

        self.temporal_aggregation = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        logger.info("Initialized VideoDeepfakeDetector with temporal aggregation")

    def forward(
        self,
        video_frames: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_frames, c, h, w = video_frames.shape

        frames_flat = video_frames.view(batch_size * num_frames, c, h, w)
        frame_features = self.frame_detector.backbone(frames_flat)

        frame_features = frame_features.view(batch_size, num_frames, -1)

        lstm_out, _ = self.temporal_aggregation(frame_features)

        aggregated_features = lstm_out[:, -1, :]

        logits = self.classifier(aggregated_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "features": aggregated_features
        }

    def predict(self, video_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        with torch.no_grad():
            outputs = self.forward(video_frames)

        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        return predictions, probabilities

    def save_model(self, save_path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'frame_model_name': self.frame_detector.model_name,
            'num_classes': 2,
            'hidden_dim': self.temporal_aggregation.hidden_size,
            'num_frames': 10
        }, save_path)
        logger.info(f"Video model saved to {save_path}")

    def load_model(self, load_path: str):
        checkpoint = torch.load(load_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Video model loaded from {load_path}")