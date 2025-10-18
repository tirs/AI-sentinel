"""
AI Sentinel - Basic Usage Examples

This script demonstrates how to use the AI Sentinel system programmatically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import HateSpeechDetector, DeepfakeDetector, MultimodalFusion
from src.data import TextPreprocessor, ImagePreprocessor
from src.explainability import SHAPExplainer, LIMEExplainer
from src.correlation import GDELTClient, EventCorrelator
from src.utils import get_logger


logger = get_logger(__name__)


def example_text_detection():
    """Example: Detect hate speech in text"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Text Detection")
    print("=" * 70)

    model = HateSpeechDetector(num_labels=3)
    preprocessor = TextPreprocessor()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    texts = [
        "This is a normal message about the weather.",
        "I hate all people from that country!",
        "You are stupid and worthless."
    ]

    encodings = preprocessor.tokenize(texts)

    # Move tensors to the same device as model
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    predictions, probabilities = model.predict(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    label_map = {0: "normal", 1: "hate", 2: "offensive"}

    for i, text in enumerate(texts):
        pred_label = label_map[predictions[i].item()]
        confidence = probabilities[i].max().item()

        print(f"\nText: {text}")
        print(f"Prediction: {pred_label.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Probabilities: Normal={probabilities[i][0]:.2%}, "
              f"Hate={probabilities[i][1]:.2%}, "
              f"Offensive={probabilities[i][2]:.2%}")


def example_text_explanation():
    """Example: Explain text predictions with LIME"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Text Explanation")
    print("=" * 70)

    model = HateSpeechDetector(num_labels=3)
    preprocessor = TextPreprocessor()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    text = "I hate all people from that country!"

    lime_explainer = LIMEExplainer(model, preprocessor.tokenizer)

    explanation = lime_explainer.explain_text(text, num_features=5)

    print(f"\nText: {text}")
    print(f"\nTop contributing words:")
    for word, weight in explanation["explanation"][:5]:
        sentiment = "POSITIVE" if weight > 0 else "NEGATIVE"
        print(f"  {word}: {weight:.3f} ({sentiment})")


def example_image_detection():
    """Example: Detect deepfakes in images"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Image Detection")
    print("=" * 70)

    model = DeepfakeDetector(num_classes=2)
    preprocessor = ImagePreprocessor(image_size=224)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)

    predictions, probabilities = model.predict(dummy_images)

    label_map = {0: "real", 1: "fake"}

    for i in range(batch_size):
        pred_label = label_map[predictions[i].item()]
        confidence = probabilities[i].max().item()

        print(f"\nImage {i + 1}:")
        print(f"Prediction: {pred_label.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Probabilities: Real={probabilities[i][0]:.2%}, "
              f"Fake={probabilities[i][1]:.2%}")


def example_multimodal_fusion():
    """Example: Combine text and image features"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Multimodal Fusion")
    print("=" * 70)

    nlp_model = HateSpeechDetector(num_labels=3)
    vision_model = DeepfakeDetector(num_classes=2)
    fusion_model = MultimodalFusion(
        text_dim=768,
        vision_dim=1280,
        num_classes=3,
        fusion_type="late"
    )

    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nlp_model = nlp_model.to(device)
    vision_model = vision_model.to(device)
    fusion_model = fusion_model.to(device)
    fusion_model.eval()

    batch_size = 2

    dummy_text_features = torch.randn(batch_size, 768).to(device)
    dummy_vision_features = torch.randn(batch_size, 1280).to(device)

    predictions, probabilities = fusion_model.predict(
        text_features=dummy_text_features,
        vision_features=dummy_vision_features
    )

    label_map = {0: "normal", 1: "violation", 2: "severe_violation"}

    for i in range(batch_size):
        pred_label = label_map[predictions[i].item()]
        confidence = probabilities[i].max().item()

        print(f"\nSample {i + 1}:")
        print(f"Prediction: {pred_label.upper()}")
        print(f"Confidence: {confidence:.2%}")


def example_gdelt_integration():
    """Example: Query GDELT for global events"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: GDELT Event Query")
    print("=" * 70)

    client = GDELTClient()

    query = "human rights protest"

    print(f"\nQuerying GDELT for: '{query}'")
    print("This may take a few seconds...")

    try:
        events = client.query_events(query, max_records=5)

        print(f"\nFound {len(events)} events:")

        for i, event in enumerate(events[:3], 1):
            print(f"\n{i}. {event.get('title', 'N/A')}")
            print(f"   URL: {event.get('url', 'N/A')}")
            print(f"   Domain: {event.get('domain', 'N/A')}")

    except Exception as e:
        print(f"\nNote: GDELT query requires internet connection")
        print(f"Error: {e}")


def example_event_correlation():
    """Example: Correlate detection with global events"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Event Correlation")
    print("=" * 70)

    from datetime import datetime

    detection = {
        "id": "test_001",
        "content": "Violent protest against government in capital city",
        "location": {"country": "Ukraine", "city": "Kyiv"},
        "timestamp": datetime.now(),
        "keywords": ["protest", "violence", "government"]
    }

    gdelt_events = [
        {
            "title": "Mass protests in Kyiv over government policies",
            "url": "https://example.com/article1",
            "domain": "example.com",
            "seendate": datetime.now().strftime("%Y%m%dT%H%M%SZ"),
            "location": {"country": "Ukraine", "city": "Kyiv"}
        },
        {
            "title": "Economic summit in Paris",
            "url": "https://example.com/article2",
            "domain": "example.com",
            "seendate": datetime.now().strftime("%Y%m%dT%H%M%SZ"),
            "location": {"country": "France", "city": "Paris"}
        }
    ]

    correlator = EventCorrelator()

    correlations = correlator.correlate_detection_with_events(
        detection,
        gdelt_events
    )

    print(f"\nDetection: {detection['content']}")
    print(f"\nFound {len(correlations)} correlated events:")

    for i, corr in enumerate(correlations, 1):
        print(f"\n{i}. {corr['event']['title']}")
        print(f"   Similarity: {corr['similarity_score']:.2%}")
        print(f"   URL: {corr['event']['url']}")


def example_preprocessing():
    """Example: Text and image preprocessing"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Data Preprocessing")
    print("=" * 70)

    text_preprocessor = TextPreprocessor()

    raw_text = "Check out this link: https://example.com @user #hashtag   "

    cleaned = text_preprocessor.clean_text(raw_text)
    language = text_preprocessor.detect_language(cleaned)

    print(f"\nOriginal: {raw_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Language: {language}")

    encodings = text_preprocessor.tokenize([cleaned])
    print(f"\nTokenized shape: {encodings['input_ids'].shape}")
    print(f"First 10 tokens: {encodings['input_ids'][0][:10].tolist()}")

    image_preprocessor = ImagePreprocessor(image_size=224)

    from PIL import Image
    import numpy as np

    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    )

    processed = image_preprocessor.preprocess_image(dummy_image)

    print(f"\nOriginal image size: 300x300")
    print(f"Processed tensor shape: {processed.shape}")
    print(f"Normalized range: [{processed.min():.2f}, {processed.max():.2f}]")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("AI SENTINEL - BASIC USAGE EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate core functionality.")
    print("Note: Some examples use dummy data for demonstration.")

    try:
        example_text_detection()
        example_text_explanation()
        example_image_detection()
        example_multimodal_fusion()
        example_gdelt_integration()
        example_event_correlation()
        example_preprocessing()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Train models: python scripts/train_nlp_model.py")
        print("2. Start API: python src/api/server.py")
        print("3. Launch dashboard: streamlit run src/dashboard/app.py")
        print("\nFor more information, see QUICKSTART.md")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\nError: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()