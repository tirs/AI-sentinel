import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

from src.config import settings, yaml_config
from src.utils import get_logger


logger = get_logger(__name__)


class GDELTClient:
    def __init__(self, api_key: Optional[str] = None):
        # GDELT provides free access - no API key required
        self.base_url = settings.GDELT_BASE_URL if hasattr(settings, 'GDELT_BASE_URL') else "https://api.gdeltproject.org/api/v2/doc/doc"
        self.correlation_config = yaml_config.get("correlation", {}).get("gdelt", {})

        logger.info("Initialized GDELT Client (Free Access)")

    def query_events(
        self,
        query: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_records: int = 250,
        mode: str = "artlist",
        format: str = "json"
    ) -> List[Dict[str, Any]]:
        # GDELT DOC 2.0 API - Use TIMESPAN format (e.g., "7d", "24h", "1w")
        # Default to last 7 days, configurable via correlation config
        lookback_days = self.correlation_config.get("lookback_days", 7)
        timespan = f"{lookback_days}d"  # Use simple format: "7d" for 7 days

        params = {
            "query": query,
            "mode": mode,
            "format": format,
            "maxrecords": max_records,
            "timespan": timespan
        }

        try:
            logger.debug(f"Making GDELT request to {self.base_url} with params: {params}")
            response = requests.get(self.base_url, params=params, timeout=30)
            logger.debug(f"Response status: {response.status_code}")

            response.raise_for_status()

            logger.debug(f"Response text length: {len(response.text)}")
            logger.debug(f"Response text (first 200 chars): {response.text[:200]}")

            data = response.json()

            articles = data.get("articles", [])
            logger.info(f"Retrieved {len(articles)} articles from GDELT for query '{query}'")

            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"GDELT API request failed: {type(e).__name__}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text[:500]}")
            return []
        except ValueError as e:
            logger.error(f"Failed to parse GDELT JSON response: {e}")
            if 'response' in locals() and response is not None:
                logger.error(f"Response status: {response.status_code}")
                logger.error(f"Response headers: {response.headers}")
                logger.error(f"Response text (full): {response.text}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in GDELT query: {type(e).__name__}: {e}")
            return []

    def query_by_location(
        self,
        location: str,
        event_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        if event_types is None:
            event_types = self.correlation_config.get("event_types", ["PROTEST", "VIOLENCE"])

        query_parts = [f"location:{location}"]

        for event_type in event_types:
            query_parts.append(f"OR {event_type}")

        query = " ".join(query_parts)

        return self.query_events(query, start_date, end_date)

    def query_by_theme(
        self,
        theme: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        query = f"theme:{theme}"
        return self.query_events(query, start_date, end_date)

    def get_trending_topics(
        self,
        location: Optional[str] = None,
        hours: int = 24
    ) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours)

        query = location if location else "*"

        articles = self.query_events(query, start_date, end_date, max_records=500)

        if not articles:
            return pd.DataFrame()

        df = pd.DataFrame(articles)

        if "seendate" in df.columns:
            df["seendate"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ")

        return df

    def correlate_with_detection(
        self,
        detection_data: Dict[str, Any],
        lookback_days: int = 7
    ) -> List[Dict[str, Any]]:
        location = detection_data.get("location")
        keywords = detection_data.get("keywords", [])
        timestamp = detection_data.get("timestamp", datetime.now())

        start_date = timestamp - timedelta(days=lookback_days)
        end_date = timestamp + timedelta(days=1)

        correlations = []

        if location:
            location_events = self.query_by_location(location, start_date=start_date, end_date=end_date)
            correlations.extend(location_events)

        for keyword in keywords[:5]:
            keyword_events = self.query_events(keyword, start_date=start_date, end_date=end_date, max_records=50)
            correlations.extend(keyword_events)

        unique_correlations = {event.get("url"): event for event in correlations}.values()

        logger.info(f"Found {len(unique_correlations)} correlated events")
        return list(unique_correlations)