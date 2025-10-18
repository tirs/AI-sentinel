import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.utils import get_logger


logger = get_logger(__name__)


class MultimodalFusion(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 1280,
        hidden_dim: int = 256,
        num_classes: int = 3,
        fusion_type: str = "late",
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        
        if fusion_type == "early":
            self.fusion_layer = nn.Sequential(
                nn.Linear(text_dim + vision_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            classifier_input_dim = hidden_dim // 2
            
        elif fusion_type == "late":
            self.text_projection = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            self.vision_projection = nn.Sequential(
                nn.Linear(vision_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            classifier_input_dim = hidden_dim * 2
            
        elif fusion_type == "attention":
            self.text_projection = nn.Linear(text_dim, hidden_dim)
            self.vision_projection = nn.Linear(vision_dim, hidden_dim)
            
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            
            classifier_input_dim = hidden_dim
            
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        logger.info(f"Initialized MultimodalFusion with {fusion_type} fusion")
    
    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if self.fusion_type == "early":
            combined = torch.cat([text_features, vision_features], dim=-1)
            fused_features = self.fusion_layer(combined)
            
        elif self.fusion_type == "late":
            text_proj = self.text_projection(text_features)
            vision_proj = self.vision_projection(vision_features)
            fused_features = torch.cat([text_proj, vision_proj], dim=-1)
            
        elif self.fusion_type == "attention":
            text_proj = self.text_projection(text_features).unsqueeze(1)
            vision_proj = self.vision_projection(vision_features).unsqueeze(1)
            
            combined = torch.cat([text_proj, vision_proj], dim=1)
            
            attended, _ = self.attention(combined, combined, combined)
            fused_features = attended.mean(dim=1)
        
        logits = self.classifier(fused_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "fused_features": fused_features
        }
    
    def predict(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(text_features, vision_features)
        
        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        
        return predictions, probabilities
    
    def save_model(self, save_path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'fusion_type': self.fusion_type,
            'num_classes': self.num_classes
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        checkpoint = torch.load(load_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {load_path}")