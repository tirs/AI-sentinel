import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json

from .models import AnalyticsRecord, ThreatDetection
from src.utils import get_logger
from src.config import get_project_root


logger = get_logger(__name__)


class DatabaseManager:
    """Thread-safe SQLite database manager for analytics and threat tracking"""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = get_project_root() / "data" / "analytics.db"

        self.db_path = db_path
        self._local = threading.local()

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize_database()

        logger.info(f"Database initialized at {self.db_path}")

    @property
    def connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA temp_store=MEMORY")
            self._local.connection.execute("PRAGMA mmap_size=268435456")  # 256MB
        return self._local.connection

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor"""
        conn = self.connection
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()

    def _initialize_database(self):
        """Create tables if they don't exist"""
        with self.get_cursor() as cursor:
            # Analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_type TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    model_accuracy REAL,
                    location TEXT,
                    language TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)

            # Threats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    threat_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    location TEXT,
                    source_content TEXT NOT NULL,
                    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    notes TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analytics (id)
                )
            """)

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_type ON analytics(analysis_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_created ON analytics(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_threats_status ON threats(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_threats_severity ON threats(severity)")

    def record_analysis(self, record: AnalyticsRecord) -> int:
        """Record an analysis and return the ID"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO analytics (
                    analysis_type, prediction, confidence, processing_time,
                    model_accuracy, location, language, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.analysis_type,
                record.prediction,
                record.confidence,
                record.processing_time,
                record.model_accuracy,
                record.location,
                record.language,
                record.created_at or datetime.now(),
                json.dumps(record.metadata) if record.metadata else None
            ))

            return cursor.lastrowid

    def record_threat(self, threat: ThreatDetection) -> int:
        """Record a threat detection and return the ID"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO threats (
                    analysis_id, threat_type, severity, confidence, status,
                    location, source_content, detected_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                threat.analysis_id,
                threat.threat_type,
                threat.severity,
                threat.confidence,
                threat.status,
                threat.location,
                threat.source_content,
                threat.detected_at or datetime.now(),
                threat.notes
            ))

            return cursor.lastrowid

    def get_analytics_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics statistics"""
        with self.get_cursor() as cursor:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Total analyses
            cursor.execute("""
                SELECT COUNT(*) FROM analytics
                WHERE created_at >= ?
            """, (cutoff_date,))
            total_analyses = cursor.fetchone()[0]

            # Previous period for comparison
            prev_cutoff = cutoff_date - timedelta(days=days)
            cursor.execute("""
                SELECT COUNT(*) FROM analytics
                WHERE created_at >= ? AND created_at < ?
            """, (prev_cutoff, cutoff_date))
            prev_analyses = cursor.fetchone()[0]

            # Calculate growth
            analyses_growth = 0
            if prev_analyses > 0:
                analyses_growth = ((total_analyses - prev_analyses) / prev_analyses) * 100

            # Average confidence/accuracy
            cursor.execute("""
                SELECT AVG(confidence) FROM analytics
                WHERE created_at >= ? AND confidence > 0
            """, (cutoff_date,))
            avg_confidence = cursor.fetchone()[0] or 0

            # Previous period confidence
            cursor.execute("""
                SELECT AVG(confidence) FROM analytics
                WHERE created_at >= ? AND created_at < ? AND confidence > 0
            """, (prev_cutoff, cutoff_date))
            prev_confidence = cursor.fetchone()[0] or 0

            # Calculate confidence growth
            confidence_growth = 0
            if prev_confidence > 0:
                confidence_growth = ((avg_confidence - prev_confidence) / prev_confidence) * 100

            # Analysis breakdown by type
            cursor.execute("""
                SELECT analysis_type, COUNT(*) FROM analytics
                WHERE created_at >= ?
                GROUP BY analysis_type
            """, (cutoff_date,))
            analysis_breakdown = dict(cursor.fetchall())

            # Average processing time
            cursor.execute("""
                SELECT AVG(processing_time) FROM analytics
                WHERE created_at >= ? AND processing_time > 0
            """, (cutoff_date,))
            avg_processing_time = cursor.fetchone()[0] or 0

            return {
                "total_analyses": total_analyses,
                "analyses_growth": analyses_growth,
                "average_confidence": avg_confidence,
                "confidence_growth": confidence_growth,
                "analysis_breakdown": analysis_breakdown,
                "average_processing_time": avg_processing_time,
                "period_days": days
            }

    def get_threat_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get threat detection statistics"""
        with self.get_cursor() as cursor:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Active threats
            cursor.execute("""
                SELECT COUNT(*) FROM threats
                WHERE detected_at >= ? AND status = 'active'
            """, (cutoff_date,))
            active_threats = cursor.fetchone()[0]

            # Previous period for comparison
            prev_cutoff = cutoff_date - timedelta(days=days)
            cursor.execute("""
                SELECT COUNT(*) FROM threats
                WHERE detected_at >= ? AND detected_at < ? AND status = 'active'
            """, (prev_cutoff, cutoff_date))
            prev_threats = cursor.fetchone()[0]

            # Calculate growth
            threats_growth = 0
            if prev_threats > 0:
                threats_growth = ((active_threats - prev_threats) / prev_threats) * 100

            # Threats by severity
            cursor.execute("""
                SELECT severity, COUNT(*) FROM threats
                WHERE detected_at >= ? AND status = 'active'
                GROUP BY severity
            """, (cutoff_date,))
            threat_severity = dict(cursor.fetchall())

            # Threats by type
            cursor.execute("""
                SELECT threat_type, COUNT(*) FROM threats
                WHERE detected_at >= ? AND status = 'active'
                GROUP BY threat_type
            """, (cutoff_date,))
            threat_types = dict(cursor.fetchall())

            # Total resolved threats
            cursor.execute("""
                SELECT COUNT(*) FROM threats
                WHERE detected_at >= ? AND status = 'resolved'
            """, (cutoff_date,))
            resolved_threats = cursor.fetchone()[0]

            return {
                "active_threats": active_threats,
                "threats_growth": threats_growth,
                "resolved_threats": resolved_threats,
                "threat_severity": threat_severity,
                "threat_types": threat_types,
                "period_days": days
            }

    def get_recent_activities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent system activities"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT
                    a.analysis_type,
                    a.prediction,
                    a.confidence,
                    a.location,
                    a.created_at,
                    t.threat_type,
                    t.severity
                FROM analytics a
                LEFT JOIN threats t ON a.id = t.analysis_id
                ORDER BY a.created_at DESC
                LIMIT ?
            """, (limit,))

            activities = []
            for row in cursor.fetchall():
                activities.append({
                    "analysis_type": row[0],
                    "prediction": row[1],
                    "confidence": row[2],
                    "location": row[3],
                    "created_at": row[4],
                    "threat_type": row[5],
                    "severity": row[6]
                })

            return activities

    def cleanup_old_records(self, days: int = 90):
        """Clean up old records to keep database size manageable"""
        with self.get_cursor() as cursor:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Delete old resolved threats
            cursor.execute("""
                DELETE FROM threats
                WHERE detected_at < ? AND status = 'resolved'
            """, (cutoff_date,))

            # Delete old analytics without associated active threats
            cursor.execute("""
                DELETE FROM analytics
                WHERE created_at < ? AND id NOT IN (
                    SELECT DISTINCT analysis_id FROM threats
                    WHERE status = 'active' AND analysis_id IS NOT NULL
                )
            """, (cutoff_date,))

            deleted_threats = cursor.rowcount
            logger.info(f"Cleaned up old records: {deleted_threats} records deleted")

    def close(self):
        """Close database connections"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection