from datetime import datetime
from typing import Dict, Any, Optional, List
import time
from contextlib import asynccontextmanager

from .manager import DatabaseManager
from .models import AnalyticsRecord, ThreatDetection
from src.utils import get_logger


logger = get_logger(__name__)


class AnalyticsService:
    """Service for handling analytics operations with performance tracking"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        logger.info("Analytics service initialized")

    @asynccontextmanager
    async def track_analysis_time(self):
        """Context manager to track analysis processing time"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            self.processing_time = end_time - start_time

    def record_text_analysis(self, text: str, prediction: str, confidence: float,
                           probabilities: Dict[str, float], language: str = None,
                           location: str = None, processing_time: float = None) -> int:
        """Record text analysis results"""

        # Determine if this is a threat
        is_threat = prediction in ['hate', 'offensive'] and confidence > 0.7

        # Create analytics record
        record = AnalyticsRecord(
            analysis_type="text",
            prediction=prediction,
            confidence=confidence,
            processing_time=processing_time or 0.0,
            model_accuracy=self._calculate_model_accuracy(probabilities),
            location=location,
            language=language,
            created_at=datetime.now(),
            metadata={
                "probabilities": probabilities,
                "text_length": len(text),
                "is_threat": is_threat
            }
        )

        analysis_id = self.db_manager.record_analysis(record)

        # Record threat if detected
        if is_threat:
            threat = ThreatDetection(
                analysis_id=analysis_id,
                threat_type="hate_speech" if prediction == "hate" else "offensive_content",
                severity=self._determine_severity(confidence),
                confidence=confidence,
                location=location,
                source_content=text[:500],  # Truncate for storage
                detected_at=datetime.now()
            )
            self.db_manager.record_threat(threat)

        logger.debug(f"Recorded text analysis: {analysis_id}, threat: {is_threat}")
        return analysis_id

    def record_image_analysis(self, filename: str, prediction: str, confidence: float,
                            probabilities: Dict[str, float], location: str = None,
                            processing_time: float = None) -> int:
        """Record image analysis results"""

        # Determine if this is a threat
        is_threat = prediction == 'fake' and confidence > 0.8

        # Create analytics record
        record = AnalyticsRecord(
            analysis_type="image",
            prediction=prediction,
            confidence=confidence,
            processing_time=processing_time or 0.0,
            model_accuracy=self._calculate_model_accuracy(probabilities),
            location=location,
            created_at=datetime.now(),
            metadata={
                "probabilities": probabilities,
                "filename": filename,
                "is_threat": is_threat
            }
        )

        analysis_id = self.db_manager.record_analysis(record)

        # Record threat if detected
        if is_threat:
            threat = ThreatDetection(
                analysis_id=analysis_id,
                threat_type="deepfake",
                severity=self._determine_severity(confidence),
                confidence=confidence,
                location=location,
                source_content=f"Image file: {filename}",
                detected_at=datetime.now()
            )
            self.db_manager.record_threat(threat)

        logger.debug(f"Recorded image analysis: {analysis_id}, threat: {is_threat}")
        return analysis_id

    def record_video_analysis(self, filename: str, prediction: str, confidence: float,
                            probabilities: Dict[str, float], frames_analyzed: int = 0,
                            location: str = None, processing_time: float = None) -> int:
        """Record video analysis results"""

        # Determine if this is a threat
        is_threat = prediction == 'fake' and confidence > 0.75

        # Create analytics record
        record = AnalyticsRecord(
            analysis_type="video",
            prediction=prediction,
            confidence=confidence,
            processing_time=processing_time or 0.0,
            model_accuracy=self._calculate_model_accuracy(probabilities),
            location=location,
            created_at=datetime.now(),
            metadata={
                "probabilities": probabilities,
                "filename": filename,
                "frames_analyzed": frames_analyzed,
                "is_threat": is_threat
            }
        )

        analysis_id = self.db_manager.record_analysis(record)

        # Record threat if detected
        if is_threat:
            threat = ThreatDetection(
                analysis_id=analysis_id,
                threat_type="video_deepfake",
                severity=self._determine_severity(confidence),
                confidence=confidence,
                location=location,
                source_content=f"Video file: {filename}",
                detected_at=datetime.now()
            )
            self.db_manager.record_threat(threat)

        logger.debug(f"Recorded video analysis: {analysis_id}, threat: {is_threat}")
        return analysis_id

    def get_dashboard_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get formatted statistics for dashboard display"""
        analytics_stats = self.db_manager.get_analytics_stats(days)
        threat_stats = self.db_manager.get_threat_stats(days)
        recent_activities = self.db_manager.get_recent_activities(10)

        return {
            # Main metrics
            "total_analyses": analytics_stats["total_analyses"],
            "analyses_growth": analytics_stats["analyses_growth"],
            "average_accuracy": analytics_stats["average_confidence"] * 100,  # Convert to percentage
            "accuracy_growth": analytics_stats["confidence_growth"],
            "active_threats": threat_stats["active_threats"],
            "threats_growth": threat_stats["threats_growth"],
            "resolved_threats": threat_stats["resolved_threats"],

            # Breakdown data
            "analysis_breakdown": analytics_stats["analysis_breakdown"],
            "threat_severity": threat_stats["threat_severity"],
            "threat_types": threat_stats["threat_types"],

            # Performance metrics
            "average_processing_time": analytics_stats["average_processing_time"],

            # Recent activities
            "recent_activities": recent_activities,

            # Period info
            "period_days": days,
            "last_updated": datetime.now().isoformat()
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        try:
            # Test database connection
            stats = self.db_manager.get_analytics_stats(1)
            db_healthy = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            db_healthy = False

        return {
            "database_healthy": db_healthy,
            "database_path": str(self.db_manager.db_path),
            "uptime": "Active",  # Could be enhanced with actual uptime tracking
            "last_cleanup": datetime.now().isoformat()  # Could track actual cleanup times
        }

    def _calculate_model_accuracy(self, probabilities: Dict[str, float]) -> float:
        """Calculate a simple accuracy metric based on prediction confidence"""
        if not probabilities:
            return 0.0

        # Use the highest probability as a proxy for accuracy
        max_prob = max(probabilities.values())

        # Adjust for multi-class scenarios
        if len(probabilities) == 2:
            # Binary classification - use max probability directly
            return max_prob
        else:
            # Multi-class - slightly penalize for more classes
            return max_prob * (0.9 + 0.1 / len(probabilities))

    def _determine_severity(self, confidence: float) -> str:
        """Determine threat severity based on confidence"""
        if confidence >= 0.95:
            return "critical"
        elif confidence >= 0.85:
            return "high"
        elif confidence >= 0.70:
            return "medium"
        else:
            return "low"

    def cleanup_old_data(self, days: int = 90):
        """Clean up old analytics data"""
        try:
            self.db_manager.cleanup_old_records(days)
            logger.info(f"Analytics cleanup completed for records older than {days} days")
        except Exception as e:
            logger.error(f"Analytics cleanup failed: {e}")

    def close(self):
        """Close analytics service"""
        if self.db_manager:
            self.db_manager.close()
            logger.info("Analytics service closed")