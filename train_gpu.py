import os
import sys

# Force use of GPU 1 (RTX A5000 - 24GB)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add project root to path
sys.path.insert(0, r'c:\Users\simba\Desktop\Ethical')

# Import and run training
from scripts.train_nlp_model import train_nlp_model

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 TRAINING NLP MODEL WITH CORRECT LABELS")
    print("=" * 80)
    print("GPU: RTX A5000 (24GB)")
    print("Dataset: UC Berkeley Measuring Hate Speech (135,556 samples)")
    print("Label Mapping:")
    print("  - NORMAL (< -1.0): Respectful, neutral content")
    print("  - OFFENSIVE (-1.0 to 1.0): Insults, profanity, disrespect")
    print("  - HATE (> 1.0): Dehumanization, violence, targeted hate")
    print("=" * 80)
    print()
    
    train_nlp_model()