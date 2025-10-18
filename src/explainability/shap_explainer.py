import torch
import numpy as np
import shap
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

from src.config import yaml_config
from src.utils import get_logger


logger = get_logger(__name__)


class SHAPExplainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Optional[AutoTokenizer] = None,
        max_samples: int = 100,
        background_samples: int = 50
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.background_samples = background_samples
        
        self.model.eval()
        
        logger.info("Initialized SHAP Explainer")
    
    def explain_text(
        self,
        texts: List[str],
        background_texts: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text explanation")
        
        def model_predict(texts_batch):
            encodings = self.tokenizer(
                texts_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                probs = torch.softmax(logits, dim=-1)
            
            return probs.cpu().numpy()
        
        if background_texts is None:
            background_texts = texts[:self.background_samples]
        
        explainer = shap.Explainer(model_predict, background_texts)
        
        shap_values = explainer(texts[:self.max_samples])
        
        explanations = []
        for i, text in enumerate(texts[:self.max_samples]):
            explanation = {
                "text": text,
                "shap_values": shap_values[i].values.tolist(),
                "base_values": shap_values[i].base_values.tolist(),
                "data": shap_values[i].data
            }
            explanations.append(explanation)
        
        logger.info(f"Generated SHAP explanations for {len(explanations)} texts")
        return explanations
    
    def explain_image(
        self,
        images: torch.Tensor,
        background_images: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        def model_predict(images_batch):
            if not isinstance(images_batch, torch.Tensor):
                images_batch = torch.tensor(images_batch, dtype=torch.float32)
            
            with torch.no_grad():
                outputs = self.model(images_batch)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                probs = torch.softmax(logits, dim=-1)
            
            return probs.cpu().numpy()
        
        if background_images is None:
            background_images = images[:self.background_samples]
        
        background_np = background_images.cpu().numpy()
        
        explainer = shap.DeepExplainer(self.model, background_images)
        
        shap_values = explainer.shap_values(images[:self.max_samples])
        
        explanations = []
        for i in range(min(len(images), self.max_samples)):
            explanation = {
                "image_index": i,
                "shap_values": shap_values[i].tolist() if isinstance(shap_values, list) else shap_values[i].tolist()
            }
            explanations.append(explanation)
        
        logger.info(f"Generated SHAP explanations for {len(explanations)} images")
        return explanations
    
    def get_top_features(
        self,
        explanation: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        shap_values = np.array(explanation["shap_values"])
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values.mean(axis=1)
        
        abs_values = np.abs(shap_values)
        top_indices = np.argsort(abs_values)[-top_k:][::-1]
        
        top_features = []
        for idx in top_indices:
            feature = {
                "index": int(idx),
                "value": float(shap_values[idx]),
                "abs_value": float(abs_values[idx])
            }
            
            if "data" in explanation and explanation["data"] is not None:
                feature["token"] = explanation["data"][idx]
            
            top_features.append(feature)
        
        return top_features