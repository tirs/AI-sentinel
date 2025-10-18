# AI Sentinel - Examples

This directory contains example scripts demonstrating how to use the AI Sentinel system.

## Available Examples

### basic_usage.py

Comprehensive examples covering:

1. **Text Detection**: Hate speech classification
2. **Text Explanation**: LIME-based explanations
3. **Image Detection**: Deepfake detection
4. **Multimodal Fusion**: Combining text and image features
5. **GDELT Integration**: Querying global events
6. **Event Correlation**: Correlating detections with events
7. **Data Preprocessing**: Text and image preprocessing

### paper_results_example.py

Examples for collecting and analyzing results for paper publication:

1. **Basic Collection**: Collect text analysis results
2. **Image Collection**: Collect image detection results
3. **GDELT Collection**: Collect global events data
4. **Generate Report**: Create markdown reports
5. **Extract Statistics**: Get quantitative metrics
6. **Create Tables**: Format results as paper tables

Perfect for gathering quantitative results for academic papers.

## Running Examples

```powershell
# Run all examples
python examples/basic_usage.py

# Or import specific examples
python -c "from examples.basic_usage import example_text_detection; example_text_detection()"
```

## Requirements

Make sure dependencies are installed:

```powershell
pip install -r requirements.txt
```

## Example Output

```
======================================================================
AI SENTINEL - BASIC USAGE EXAMPLES
======================================================================

EXAMPLE 1: Text Detection
======================================================================

Text: This is a normal message about the weather.
Prediction: NORMAL
Confidence: 85.23%
Probabilities: Normal=85.23%, Hate=7.45%, Offensive=7.32%

...
```

## Next Steps

After running examples:

1. **Train Models**: `python scripts/train_nlp_model.py`
2. **Start API**: `python src/api/server.py`
3. **Launch Dashboard**: `streamlit run src/dashboard/app.py`

## Custom Examples

Create your own examples by importing from `src`:

```python
from src.models import HateSpeechDetector
from src.data import TextPreprocessor

model = HateSpeechDetector()
preprocessor = TextPreprocessor()

# Your code here
```

## Documentation

- **QUICKSTART.md**: Installation guide
- **ARCHITECTURE.md**: Technical details
- **API Docs**: http://localhost:8000/docs (when API is running)