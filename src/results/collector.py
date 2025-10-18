"""
Results collection system for automated data gathering from analysis modules.
Collects results from text, image, video, GDELT, and multimodal analyses.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import aiohttp
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for individual analysis result"""
    module: str  # text, image, video, gdelt, multimodal
    input_id: str
    input_source: str  # text content, image path, video path, query, etc.
    prediction: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    explanation: Optional[Dict[str, Any]] = None
    correlations: Optional[List[Dict[str, Any]]] = None
    processing_time: float = 0.0


class ResultsCollector:
    """Collects results from all analysis modules"""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.results: List[AnalysisResult] = []
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _serialize_result(self, result: AnalysisResult) -> Dict[str, Any]:
        """Convert result to serializable format"""
        data = asdict(result)
        data['timestamp'] = result.timestamp.isoformat()
        return data

    async def collect_text_analysis(
        self,
        texts: List[str],
        explain: bool = True,
        correlate: bool = False,
        location: Optional[str] = None
    ) -> List[AnalysisResult]:
        """Collect text detection analysis results"""
        results = []

        for idx, text in enumerate(texts):
            try:
                start_time = datetime.now()

                payload = {
                    "text": text,
                    "explain": explain,
                    "correlate": correlate,
                    "location": location
                }

                async with self.session.post(
                    f"{self.api_base_url}/detect/text",
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        processing_time = (datetime.now() - start_time).total_seconds()

                        result = AnalysisResult(
                            module="text",
                            input_id=f"text_{idx}",
                            input_source=text[:100],
                            prediction=data.get("prediction", "unknown"),
                            confidence=data.get("confidence", 0.0),
                            timestamp=datetime.now(),
                            metadata={
                                "language": data.get("language"),
                                "probabilities": data.get("probabilities")
                            },
                            explanation=data.get("explanation"),
                            correlations=data.get("correlations"),
                            processing_time=processing_time
                        )
                        results.append(result)
                        self.results.append(result)
                    else:
                        logger.error(f"Text analysis failed: {response.status}")

            except Exception as e:
                logger.error(f"Error collecting text analysis for {idx}: {e}")

        return results

    async def collect_image_analysis(
        self,
        image_paths: List[str],
        explain: bool = True
    ) -> List[AnalysisResult]:
        """Collect image detection analysis results"""
        results = []

        for idx, image_path in enumerate(image_paths):
            try:
                start_time = datetime.now()
                path = Path(image_path)

                if not path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    continue

                with open(image_path, 'rb') as f:
                    form_data = aiohttp.FormData()
                    form_data.add_field('file', f, filename=path.name)
                    form_data.add_field('explain', str(explain).lower())

                    async with self.session.post(
                        f"{self.api_base_url}/detect/image",
                        data=form_data,
                        timeout=60
                    ) as response:
                        if response.status == 200:
                            result_data = await response.json()
                            processing_time = (datetime.now() - start_time).total_seconds()

                            result = AnalysisResult(
                                module="image",
                                input_id=f"image_{idx}",
                                input_source=path.name,
                                prediction=result_data.get("prediction", "unknown"),
                                confidence=result_data.get("confidence", 0.0),
                                timestamp=datetime.now(),
                                metadata={
                                    "probabilities": result_data.get("probabilities"),
                                    "image_path": str(image_path)
                                },
                                explanation=result_data.get("explanation"),
                                processing_time=processing_time
                            )
                            results.append(result)
                            self.results.append(result)
                        else:
                            logger.error(f"Image analysis failed: {response.status}")

            except Exception as e:
                logger.error(f"Error collecting image analysis for {image_path}: {e}")

        return results

    async def collect_video_analysis(
        self,
        video_paths: List[str],
        frame_interval: int = 10,
        max_frames: int = 30
    ) -> List[AnalysisResult]:
        """Collect video detection analysis results"""
        results = []

        for idx, video_path in enumerate(video_paths):
            try:
                start_time = datetime.now()
                path = Path(video_path)

                if not path.exists():
                    logger.warning(f"Video not found: {video_path}")
                    continue

                with open(video_path, 'rb') as f:
                    form_data = aiohttp.FormData()
                    form_data.add_field('file', f, filename=path.name)
                    form_data.add_field('frame_interval', str(frame_interval))
                    form_data.add_field('max_frames', str(max_frames))

                    async with self.session.post(
                        f"{self.api_base_url}/detect/video",
                        data=form_data,
                        timeout=120
                    ) as response:
                        if response.status == 200:
                            result_data = await response.json()
                            processing_time = (datetime.now() - start_time).total_seconds()

                            result = AnalysisResult(
                                module="video",
                                input_id=f"video_{idx}",
                                input_source=path.name,
                                prediction=result_data.get("prediction", "unknown"),
                                confidence=result_data.get("confidence", 0.0),
                                timestamp=datetime.now(),
                                metadata={
                                    "frames_analyzed": result_data.get("frames_analyzed"),
                                    "statistics": result_data.get("statistics"),
                                    "frame_predictions": result_data.get("frame_predictions"),
                                    "frame_confidences": result_data.get("frame_confidences"),
                                    "video_path": str(video_path)
                                },
                                processing_time=processing_time
                            )
                            results.append(result)
                            self.results.append(result)
                        else:
                            logger.error(f"Video analysis failed: {response.status}")

            except Exception as e:
                logger.error(f"Error collecting video analysis for {video_path}: {e}")

        return results

    async def collect_gdelt_analysis(
        self,
        queries: List[str],
        max_records: int = 100,
        time_range_days: int = 7
    ) -> List[AnalysisResult]:
        """Collect GDELT event analysis results"""
        results = []

        for idx, query in enumerate(queries):
            try:
                start_time = datetime.now()

                params = {
                    "query": query,
                    "max_records": max_records
                }

                async with self.session.get(
                    f"{self.api_base_url}/gdelt/events",
                    params=params,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        processing_time = (datetime.now() - start_time).total_seconds()
                        events = result_data.get("events", [])
                        event_count = result_data.get("count", len(events))

                        result = AnalysisResult(
                            module="gdelt",
                            input_id=f"gdelt_{idx}",
                            input_source=query,
                            prediction="success",
                            confidence=float(len(events) > 0),
                            timestamp=datetime.now(),
                            metadata={
                                "query": query,
                                "total_events": event_count,
                                "events": events[:50] if events else [],
                                "max_records": max_records
                            },
                            processing_time=processing_time
                        )
                        results.append(result)
                        self.results.append(result)
                    else:
                        logger.error(f"GDELT analysis failed: {response.status}")

            except Exception as e:
                logger.error(f"Error collecting GDELT analysis for query '{query}': {e}")

        return results

    def save_results(self, output_path: Path) -> None:
        """Save collected results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "collection_timestamp": datetime.now().isoformat(),
            "total_results": len(self.results),
            "results": [self._serialize_result(r) for r in self.results]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def get_results_by_module(self, module: str) -> List[AnalysisResult]:
        """Filter results by module"""
        return [r for r in self.results if r.module == module]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all results"""
        return {
            "total_results": len(self.results),
            "modules": list(set(r.module for r in self.results)),
            "by_module": {
                module: {
                    "count": len(self.get_results_by_module(module)),
                    "avg_confidence": sum(
                        r.confidence for r in self.get_results_by_module(module)
                    ) / max(len(self.get_results_by_module(module)), 1),
                    "avg_processing_time": sum(
                        r.processing_time for r in self.get_results_by_module(module)
                    ) / max(len(self.get_results_by_module(module)), 1)
                }
                for module in set(r.module for r in self.results)
            }
        }