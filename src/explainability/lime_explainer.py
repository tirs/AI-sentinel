import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer
from typing import List, Dict, Any, Callable, Optional
from transformers import AutoTokenizer

from src.config import yaml_config
from src.utils import get_logger


logger = get_logger(__name__)


class LIMEExplainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Optional[AutoTokenizer] = None,
        num_features: int = 10,
        num_samples: int = 1000
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_features = num_features
        self.num_samples = num_samples

        self.model.eval()

        self.text_explainer = LimeTextExplainer(
            class_names=['normal', 'hate', 'offensive']
        )

        self.image_explainer = LimeImageExplainer()

        logger.info("Initialized LIME Explainer")

    def explain_text(
        self,
        text: str,
        num_features: Optional[int] = None
    ) -> Dict[str, Any]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text explanation")

        num_features = num_features or self.num_features

        def predict_proba(texts: List[str]) -> np.ndarray:
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move tensors to the same device as the model
            device = next(self.model.parameters()).device
            encodings = {k: v.to(device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                probs = torch.softmax(logits, dim=-1)

            return probs.cpu().numpy()

        explanation = self.text_explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features,
            num_samples=self.num_samples
        )

        # Convert numpy types to Python native types for JSON serialization
        local_pred = explanation.local_pred
        if hasattr(local_pred, 'tolist'):
            local_pred = local_pred.tolist()
        elif isinstance(local_pred, (np.integer, np.floating)):
            local_pred = local_pred.item()

        result = {
            "text": text,
            "explanation": explanation.as_list(),
            "prediction_probabilities": explanation.predict_proba.tolist(),
            "local_prediction": local_pred
        }

        logger.info(f"Generated LIME explanation for text")
        return result

    def explain_image(
        self,
        image: np.ndarray,
        num_features: Optional[int] = None,
        hide_color: int = 0
    ) -> Dict[str, Any]:
        num_features = num_features or self.num_features

        def predict_proba(images: np.ndarray) -> np.ndarray:
            images_tensor = torch.tensor(images, dtype=torch.float32)

            if images_tensor.dim() == 3:
                images_tensor = images_tensor.unsqueeze(0)

            if images_tensor.shape[-1] == 3:
                images_tensor = images_tensor.permute(0, 3, 1, 2)

            # Move tensor to the same device as the model
            device = next(self.model.parameters()).device
            images_tensor = images_tensor.to(device)

            with torch.no_grad():
                outputs = self.model(images_tensor)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                probs = torch.softmax(logits, dim=-1)

            return probs.cpu().numpy()

        explanation = self.image_explainer.explain_instance(
            image,
            predict_proba,
            top_labels=3,
            hide_color=hide_color,
            num_samples=self.num_samples
        )

        # Convert numpy types to Python native types for JSON serialization
        top_labels = explanation.top_labels
        if isinstance(top_labels, np.ndarray):
            top_labels = top_labels.tolist()
        elif isinstance(top_labels, (list, tuple)):
            top_labels = [int(x) if isinstance(x, (np.integer, np.int64)) else x for x in top_labels]

        local_pred = explanation.local_pred
        if isinstance(local_pred, np.ndarray):
            local_pred = local_pred.tolist()
        elif isinstance(local_pred, (np.integer, np.floating)):
            local_pred = local_pred.item()

        result = {
            "top_labels": top_labels,
            "local_prediction": local_pred,
            "segments": explanation.segments.tolist()
        }

        logger.info(f"Generated LIME explanation for image")
        return result

    def get_text_highlights(
        self,
        explanation: Dict[str, Any],
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        highlights = []

        for word, weight in explanation["explanation"]:
            if abs(weight) >= threshold:
                highlights.append({
                    "word": word,
                    "weight": weight,
                    "sentiment": "positive" if weight > 0 else "negative"
                })

        return highlights