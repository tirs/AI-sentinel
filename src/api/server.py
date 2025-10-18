from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from datetime import datetime

from src.config import settings, yaml_config, get_project_root
from src.models import HateSpeechDetector, DeepfakeDetector, VideoDeepfakeDetector, MultimodalFusion
from src.data import TextPreprocessor, ImagePreprocessor
from src.explainability import SHAPExplainer, LIMEExplainer
from src.correlation import GDELTClient, EventCorrelator
from src.utils import get_logger


logger = get_logger(__name__)

app = FastAPI(
    title="AI Sentinel API",
    description="Multimodal Explainable System for Detecting Digital Human Rights Violations",
    version="1.0.0"
)

api_config = yaml_config.get("api", {})
cors_origins = api_config.get("cors_origins", ["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextDetectionRequest(BaseModel):
    text: str
    explain: bool = False
    correlate: bool = False
    location: Optional[str] = None


class TextDetectionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    language: str
    explanation: Optional[Dict[str, Any]] = None
    correlations: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]


nlp_model = None
vision_model = None
video_model = None
text_preprocessor = None
image_preprocessor = None
gdelt_client = None
event_correlator = None


@app.on_event("startup")
async def startup_event():
    global nlp_model, vision_model, video_model, text_preprocessor, image_preprocessor
    global gdelt_client, event_correlator

    logger.info("Starting AI Sentinel API...")

    try:
        # Initialize preprocessors
        nlp_config = yaml_config["models"]["nlp"]
        vision_config = yaml_config["models"]["vision"]

        text_preprocessor = TextPreprocessor(nlp_config["base_model"])
        image_preprocessor = ImagePreprocessor(
            image_size=vision_config["image_size"],
            augmentation=vision_config["augmentation"]
        )

        # Load NLP model if available
        nlp_model_path = get_project_root() / settings.MODELS_DIR / "hate_speech_detector.pth"
        if nlp_model_path.exists():
            logger.info(f"Loading NLP model from {nlp_model_path}")
            nlp_model = HateSpeechDetector(
                model_name=nlp_config["base_model"],
                num_labels=nlp_config["num_labels"]
            )
            nlp_model.load_model(str(nlp_model_path))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            nlp_model = nlp_model.to(device)
            nlp_model.eval()
            logger.info("NLP model loaded successfully")
        else:
            logger.warning(f"NLP model not found at {nlp_model_path}. Text detection will be unavailable.")

        # Load Vision model if available
        vision_model_path = get_project_root() / settings.MODELS_DIR / "deepfake_detector.pth"
        if vision_model_path.exists():
            logger.info(f"Loading Vision model from {vision_model_path}")
            vision_model = DeepfakeDetector(
                model_name=vision_config["base_model"],
                num_classes=vision_config["num_classes"]
            )
            vision_model.load_model(str(vision_model_path))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vision_model = vision_model.to(device)
            vision_model.eval()
            logger.info("Vision model loaded successfully")
        else:
            logger.warning(f"Vision model not found at {vision_model_path}. Image detection will be unavailable.")

        # Load Video model if available (uses vision model as base)
        video_model_path = get_project_root() / settings.MODELS_DIR / "video_deepfake_detector.pth"
        if video_model_path.exists():
            logger.info(f"Loading Video model from {video_model_path}")
            video_model = VideoDeepfakeDetector(
                frame_model_name=vision_config["base_model"],
                num_classes=vision_config["num_classes"]
            )
            video_model.load_model(str(video_model_path))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            video_model = video_model.to(device)
            video_model.eval()
            logger.info("Video model loaded successfully")
        else:
            logger.warning(f"Video model not found at {video_model_path}. Using frame-by-frame detection with vision model.")

        # Initialize GDELT client and correlator
        gdelt_client = GDELTClient()
        event_correlator = EventCorrelator(gdelt_client)

        logger.info("API startup complete")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "nlp": nlp_model is not None,
            "vision": vision_model is not None,
            "video": video_model is not None or vision_model is not None,
            "preprocessors": text_preprocessor is not None
        }
    )


@app.post("/detect/text", response_model=TextDetectionResponse)
async def detect_text(request: TextDetectionRequest):
    try:
        if nlp_model is None:
            raise HTTPException(status_code=503, detail="NLP model not loaded")

        language = text_preprocessor.detect_language(request.text)

        encodings = text_preprocessor.tokenize([request.text])

        # Move tensors to the same device as the model
        device = next(nlp_model.parameters()).device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        predictions, probabilities = nlp_model.predict(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        label_map = {0: "normal", 1: "hate", 2: "offensive"}
        prediction_label = label_map[predictions[0].item()]
        confidence = probabilities[0].max().item()

        prob_dict = {
            label: float(probabilities[0][idx])
            for idx, label in label_map.items()
        }

        explanation = None
        if request.explain:
            lime_explainer = LIMEExplainer(nlp_model, text_preprocessor.tokenizer)
            explanation = lime_explainer.explain_text(request.text)

        correlations = None
        if request.correlate and request.location:
            detection_data = {
                "id": f"text_{datetime.now().timestamp()}",
                "content": request.text,
                "location": request.location,
                "timestamp": datetime.now()
            }

            gdelt_events = gdelt_client.query_by_location(request.location)
            correlations = event_correlator.correlate_detection_with_events(
                detection_data,
                gdelt_events
            )

        return TextDetectionResponse(
            prediction=prediction_label,
            confidence=confidence,
            probabilities=prob_dict,
            language=language,
            explanation=explanation,
            correlations=correlations,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Text detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    explain: bool = False
):
    try:
        if vision_model is None:
            raise HTTPException(status_code=503, detail="Vision model not loaded")

        contents = await file.read()

        from PIL import Image
        import io

        image = Image.open(io.BytesIO(contents))

        image_tensor = image_preprocessor.preprocess_image(image).unsqueeze(0)

        # Move tensor to the same device as the model
        device = next(vision_model.parameters()).device
        image_tensor = image_tensor.to(device)

        predictions, probabilities = vision_model.predict(image_tensor)

        label_map = {0: "real", 1: "fake"}
        prediction_label = label_map[predictions[0].item()]
        confidence = probabilities[0].max().item()

        response = {
            "prediction": prediction_label,
            "confidence": float(confidence),
            "probabilities": {
                "real": float(probabilities[0][0]),
                "fake": float(probabilities[0][1])
            },
            "timestamp": datetime.now().isoformat()
        }

        if explain:
            lime_explainer = LIMEExplainer(vision_model)
            import numpy as np
            image_np = np.array(image)
            explanation = lime_explainer.explain_image(image_np)
            response["explanation"] = explanation

        return response

    except Exception as e:
        logger.error(f"Image detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    frame_interval: int = 10,
    max_frames: int = 30
):
    try:
        if vision_model is None:
            raise HTTPException(status_code=503, detail="Vision model not loaded")

        import tempfile
        import os
        from PIL import Image
        import io

        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_video_path = tmp_file.name

        try:
            # Extract frames from video
            frames = image_preprocessor.extract_video_frames(
                tmp_video_path,
                num_frames=max_frames,
                method="uniform"
            )

            if not frames:
                raise HTTPException(status_code=400, detail="No frames could be extracted from video")

            # Process frames
            device = next(vision_model.parameters()).device
            frame_predictions = []
            frame_confidences = []

            # If we have a dedicated video model, use it
            if video_model is not None:
                # Preprocess all frames for video model
                processed_frames = [image_preprocessor.preprocess_video_frame(frame) for frame in frames]
                video_tensor = torch.stack(processed_frames).unsqueeze(0)  # Add batch dimension
                video_tensor = video_tensor.to(device)

                predictions, probabilities = video_model.predict(video_tensor)

                label_map = {0: "real", 1: "fake"}
                prediction_label = label_map[predictions[0].item()]
                confidence = probabilities[0].max().item()

                # Also get per-frame predictions for visualization
                for frame in frames:
                    frame_tensor = image_preprocessor.preprocess_video_frame(frame).unsqueeze(0).to(device)
                    frame_pred, frame_prob = vision_model.predict(frame_tensor)
                    frame_predictions.append(label_map[frame_pred[0].item()])
                    frame_confidences.append(float(frame_prob[0].max().item()))

            else:
                # Fall back to frame-by-frame detection with vision model
                for frame in frames:
                    frame_tensor = image_preprocessor.preprocess_video_frame(frame).unsqueeze(0).to(device)
                    frame_pred, frame_prob = vision_model.predict(frame_tensor)

                    label_map = {0: "real", 1: "fake"}
                    frame_predictions.append(label_map[frame_pred[0].item()])
                    frame_confidences.append(float(frame_prob[0].max().item()))

                # Aggregate predictions (majority vote)
                fake_count = frame_predictions.count("fake")
                real_count = frame_predictions.count("real")
                prediction_label = "fake" if fake_count > real_count else "real"
                confidence = max(fake_count, real_count) / len(frame_predictions)

            # Calculate statistics
            avg_confidence = sum(frame_confidences) / len(frame_confidences)
            fake_frame_percentage = (frame_predictions.count("fake") / len(frame_predictions)) * 100

            response = {
                "prediction": prediction_label,
                "confidence": float(confidence),
                "probabilities": {
                    "real": 1.0 - confidence if prediction_label == "fake" else confidence,
                    "fake": confidence if prediction_label == "fake" else 1.0 - confidence
                },
                "frames_analyzed": len(frames),
                "frame_predictions": frame_predictions,
                "frame_confidences": frame_confidences,
                "statistics": {
                    "avg_confidence": float(avg_confidence),
                    "fake_frames": frame_predictions.count("fake"),
                    "real_frames": frame_predictions.count("real"),
                    "fake_percentage": float(fake_frame_percentage)
                },
                "model_type": "temporal" if video_model is not None else "frame_by_frame",
                "timestamp": datetime.now().isoformat()
            }

            return response

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)

    except Exception as e:
        logger.error(f"Video detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gdelt/events")
async def get_gdelt_events(
    query: str,
    max_records: int = 100
):
    try:
        events = gdelt_client.query_events(query, max_records=max_records)

        return {
            "query": query,
            "count": len(events),
            "events": events
        }

    except Exception as e:
        logger.error(f"GDELT query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gdelt/trending")
async def get_trending_topics(
    location: Optional[str] = None,
    hours: int = 24
):
    try:
        df = gdelt_client.get_trending_topics(location, hours)

        return {
            "location": location or "global",
            "hours": hours,
            "topics": df.to_dict(orient="records") if not df.empty else []
        }

    except Exception as e:
        logger.error(f"Trending topics query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=False
    )