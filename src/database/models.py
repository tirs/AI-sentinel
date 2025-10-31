from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import json


@dataclass
class AnalyticsRecord:
    """Record for tracking system analytics"""
    id: Optional[int] = None
    analysis_type: str = ""  # 'text', 'image', 'video'
    prediction: str = ""  # 'hate', 'normal', 'fake', 'real', etc.
    confidence: float = 0.0
    processing_time: float = 0.0  # seconds
    model_accuracy: Optional[float] = None
    location: Optional[str] = None
    language: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "analysis_type": self.analysis_type,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "model_accuracy": self.model_accuracy,
            "location": self.location,
            "language": self.language,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": json.dumps(self.metadata) if self.metadata else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalyticsRecord":
        """Create from dictionary"""
        return cls(
            id=data.get("id"),
            analysis_type=data.get("analysis_type", ""),
            prediction=data.get("prediction", ""),
            confidence=data.get("confidence", 0.0),
            processing_time=data.get("processing_time", 0.0),
            model_accuracy=data.get("model_accuracy"),
            location=data.get("location"),
            language=data.get("language"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            metadata=json.loads(data["metadata"]) if data.get("metadata") else None
        )


@dataclass
class ThreatDetection:
    """Record for threat detections"""
    id: Optional[int] = None
    analysis_id: Optional[int] = None  # Foreign key to AnalyticsRecord
    threat_type: str = ""  # 'hate_speech', 'deepfake', 'misinformation', etc.
    severity: str = ""  # 'low', 'medium', 'high', 'critical'
    confidence: float = 0.0
    status: str = "active"  # 'active', 'resolved', 'false_positive'
    location: Optional[str] = None
    source_content: str = ""
    detected_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "threat_type": self.threat_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "status": self.status,
            "location": self.location,
            "source_content": self.source_content,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreatDetection":
        """Create from dictionary"""
        return cls(
            id=data.get("id"),
            analysis_id=data.get("analysis_id"),
            threat_type=data.get("threat_type", ""),
            severity=data.get("severity", ""),
            confidence=data.get("confidence", 0.0),
            status=data.get("status", "active"),
            location=data.get("location"),
            source_content=data.get("source_content", ""),
            detected_at=datetime.fromisoformat(data["detected_at"]) if data.get("detected_at") else None,
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            notes=data.get("notes")
        )