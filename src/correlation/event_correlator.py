from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.correlation.gdelt_client import GDELTClient
from src.config import yaml_config
from src.utils import get_logger


logger = get_logger(__name__)


class EventCorrelator:
    def __init__(self, gdelt_client: Optional[GDELTClient] = None):
        self.gdelt_client = gdelt_client or GDELTClient()
        self.correlation_config = yaml_config.get("correlation", {}).get("gdelt", {})
        self.min_confidence = self.correlation_config.get("min_confidence", 0.7)
        
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        logger.info("Initialized Event Correlator")
    
    def correlate_detection_with_events(
        self,
        detection: Dict[str, Any],
        gdelt_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not gdelt_events:
            return []
        
        detection_text = self._extract_detection_text(detection)
        event_texts = [self._extract_event_text(event) for event in gdelt_events]
        
        all_texts = [detection_text] + event_texts
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            detection_vector = tfidf_matrix[0:1]
            event_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(detection_vector, event_vectors)[0]
            
            correlations = []
            for idx, similarity in enumerate(similarities):
                if similarity >= self.min_confidence:
                    correlation = {
                        "event": gdelt_events[idx],
                        "similarity_score": float(similarity),
                        "detection_id": detection.get("id"),
                        "correlation_type": "content_similarity"
                    }
                    correlations.append(correlation)
            
            correlations.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            logger.info(f"Found {len(correlations)} correlated events with confidence >= {self.min_confidence}")
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation failed: {e}")
            return []
    
    def correlate_temporal(
        self,
        detection: Dict[str, Any],
        gdelt_events: List[Dict[str, Any]],
        time_window_hours: int = 48
    ) -> List[Dict[str, Any]]:
        detection_time = detection.get("timestamp")
        
        if isinstance(detection_time, str):
            detection_time = datetime.fromisoformat(detection_time)
        elif detection_time is None:
            detection_time = datetime.now()
        
        time_window = timedelta(hours=time_window_hours)
        
        temporal_correlations = []
        
        for event in gdelt_events:
            event_time = event.get("seendate")
            
            if isinstance(event_time, str):
                try:
                    event_time = datetime.strptime(event_time, "%Y%m%dT%H%M%SZ")
                except ValueError:
                    continue
            
            if event_time is None:
                continue
            
            time_diff = abs((detection_time - event_time).total_seconds() / 3600)
            
            if time_diff <= time_window_hours:
                temporal_score = 1.0 - (time_diff / time_window_hours)
                
                correlation = {
                    "event": event,
                    "temporal_score": temporal_score,
                    "time_difference_hours": time_diff,
                    "detection_id": detection.get("id"),
                    "correlation_type": "temporal"
                }
                temporal_correlations.append(correlation)
        
        temporal_correlations.sort(key=lambda x: x["temporal_score"], reverse=True)
        
        logger.info(f"Found {len(temporal_correlations)} temporal correlations")
        return temporal_correlations
    
    def correlate_spatial(
        self,
        detection: Dict[str, Any],
        gdelt_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        detection_location = detection.get("location", {})
        
        if not detection_location:
            return []
        
        spatial_correlations = []
        
        for event in gdelt_events:
            event_location = self._extract_event_location(event)
            
            if not event_location:
                continue
            
            spatial_score = self._calculate_spatial_similarity(detection_location, event_location)
            
            if spatial_score > 0:
                correlation = {
                    "event": event,
                    "spatial_score": spatial_score,
                    "detection_id": detection.get("id"),
                    "correlation_type": "spatial"
                }
                spatial_correlations.append(correlation)
        
        spatial_correlations.sort(key=lambda x: x["spatial_score"], reverse=True)
        
        logger.info(f"Found {len(spatial_correlations)} spatial correlations")
        return spatial_correlations
    
    def multi_dimensional_correlation(
        self,
        detection: Dict[str, Any],
        gdelt_events: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        if weights is None:
            weights = {
                "content": 0.4,
                "temporal": 0.3,
                "spatial": 0.3
            }
        
        content_corr = self.correlate_detection_with_events(detection, gdelt_events)
        temporal_corr = self.correlate_temporal(detection, gdelt_events)
        spatial_corr = self.correlate_spatial(detection, gdelt_events)
        
        event_scores = {}
        
        for corr in content_corr:
            event_url = corr["event"].get("url")
            if event_url:
                event_scores[event_url] = {
                    "event": corr["event"],
                    "content_score": corr["similarity_score"],
                    "temporal_score": 0.0,
                    "spatial_score": 0.0
                }
        
        for corr in temporal_corr:
            event_url = corr["event"].get("url")
            if event_url:
                if event_url in event_scores:
                    event_scores[event_url]["temporal_score"] = corr["temporal_score"]
                else:
                    event_scores[event_url] = {
                        "event": corr["event"],
                        "content_score": 0.0,
                        "temporal_score": corr["temporal_score"],
                        "spatial_score": 0.0
                    }
        
        for corr in spatial_corr:
            event_url = corr["event"].get("url")
            if event_url:
                if event_url in event_scores:
                    event_scores[event_url]["spatial_score"] = corr["spatial_score"]
                else:
                    event_scores[event_url] = {
                        "event": corr["event"],
                        "content_score": 0.0,
                        "temporal_score": 0.0,
                        "spatial_score": corr["spatial_score"]
                    }
        
        final_correlations = []
        for event_url, scores in event_scores.items():
            combined_score = (
                weights["content"] * scores["content_score"] +
                weights["temporal"] * scores["temporal_score"] +
                weights["spatial"] * scores["spatial_score"]
            )
            
            correlation = {
                "event": scores["event"],
                "combined_score": combined_score,
                "content_score": scores["content_score"],
                "temporal_score": scores["temporal_score"],
                "spatial_score": scores["spatial_score"],
                "detection_id": detection.get("id"),
                "correlation_type": "multi_dimensional"
            }
            final_correlations.append(correlation)
        
        final_correlations.sort(key=lambda x: x["combined_score"], reverse=True)
        
        logger.info(f"Generated {len(final_correlations)} multi-dimensional correlations")
        return final_correlations
    
    def _extract_detection_text(self, detection: Dict[str, Any]) -> str:
        text_parts = []
        
        if "content" in detection:
            text_parts.append(detection["content"])
        
        if "keywords" in detection:
            text_parts.extend(detection["keywords"])
        
        if "description" in detection:
            text_parts.append(detection["description"])
        
        return " ".join(text_parts)
    
    def _extract_event_text(self, event: Dict[str, Any]) -> str:
        text_parts = []
        
        if "title" in event:
            text_parts.append(event["title"])
        
        if "description" in event:
            text_parts.append(event["description"])
        
        return " ".join(text_parts)
    
    def _extract_event_location(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "location" in event:
            return event["location"]
        
        return None
    
    def _calculate_spatial_similarity(
        self,
        loc1: Dict[str, Any],
        loc2: Dict[str, Any]
    ) -> float:
        if "country" in loc1 and "country" in loc2:
            if loc1["country"] == loc2["country"]:
                return 1.0
        
        if "city" in loc1 and "city" in loc2:
            if loc1["city"] == loc2["city"]:
                return 1.0
        
        return 0.0