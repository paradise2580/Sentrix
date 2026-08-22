"""
src/ingestion/scraper.py

Role
----
Fetches live external signals from free-tier APIs. Each source is its own
method so they can be tested, scheduled, and rate-limited independently
(this module is what the Airflow DAG calls every 6 hours).

Design note
-----------
This module ONLY fetches and returns raw signals. It does not clean them
or turn them into model features — that is src/preprocessing's job. Keeping
fetch and transform separate means a failed API call never corrupts
feature logic downstream.

No-key behaviour
-----------------
If an API key is missing from .env, each method logs a warning and returns
an empty result rather than raising — a missing key should degrade the
pipeline gracefully, not crash it. Once real keys are added to .env, these
same methods start returning live data with no code changes required.
"""

import os
import requests
from datetime import date
from dotenv import load_dotenv

from src.logger import get_logger
from src.exception import SentrixException
import sys

load_dotenv()
logger = get_logger(__name__)

_NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
_OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "")
_FRED_API_KEY = os.getenv("FRED_API_KEY", "")


class SignalScraper:
    """Fetches one external signal type per public method."""

    def fetch_news(self, region: str, page_size: int = 10) -> list[dict]:
        """
        Recent news headlines for a supplier's region via NewsAPI.
        Returns a list of {title, description, published_at, source}.
        """
        if not _NEWSAPI_KEY:
            logger.warning("NEWSAPI_KEY not set — skipping live news fetch for '%s'", region)
            return []
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": region,
                    "pageSize": page_size,
                    "sortBy": "publishedAt",
                    "apiKey": _NEWSAPI_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            logger.info(f"Fetched {len(articles)} news articles for region='{region}'")
            return [
                {
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "published_at": a.get("publishedAt"),
                    "source": a.get("source", {}).get("name"),
                }
                for a in articles
            ]
        except Exception as e:
            raise SentrixException(e, sys)

    def fetch_weather(self, lat: float, lon: float) -> dict | None:
        """Current weather severity signal for a supplier's coordinates via OpenWeatherMap."""
        if not _OPENWEATHER_KEY:
            logger.warning("OPENWEATHER_KEY not set — skipping live weather fetch")
            return None
        try:
            resp = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": lat, "lon": lon, "appid": _OPENWEATHER_KEY, "units": "metric"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "condition": data.get("weather", [{}])[0].get("main"),
                "wind_speed": data.get("wind", {}).get("speed"),
                "fetched_at": date.today().isoformat(),
            }
        except Exception as e:
            raise SentrixException(e, sys)

    def fetch_commodity(self, series_id: str = "PPIACO") -> list[dict]:
        """Commodity price series via the FRED API (US Federal Reserve, free)."""
        if not _FRED_API_KEY:
            logger.warning("FRED_API_KEY not set — skipping live commodity fetch")
            return []
        try:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": _FRED_API_KEY,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 30,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("observations", [])
        except Exception as e:
            raise SentrixException(e, sys)

    def fetch_port_status(self) -> dict | None:
        """
        Port congestion index. MarineTraffic's free tier is heavily
        restricted, so this is a placeholder integration point — swap in
        a real port-data provider's endpoint here when available.
        """
        logger.warning("Port status live fetch not configured — using synthetic fallback upstream")
        return None

    def fetch_all_signals(self, suppliers: list[dict]) -> dict:
        """
        Fetch every signal type for a list of suppliers, e.g.
        [{"supplier_id": 1, "region": "Shanghai, China", "lat": 31.23, "lon": 121.47}, ...]
        This is the entry point the Airflow DAG calls.
        """
        results = {"news": [], "weather": [], "commodity": [], "port": []}
        for s in suppliers:
            results["news"].append(self.fetch_news(s["region"]))
            if "lat" in s and "lon" in s:
                results["weather"].append(self.fetch_weather(s["lat"], s["lon"]))
        results["commodity"] = self.fetch_commodity()
        results["port"] = self.fetch_port_status()
        return results


if __name__ == "__main__":
    scraper = SignalScraper()
    print("News (empty if NEWSAPI_KEY unset):", scraper.fetch_news("Shanghai, China"))
    print("Weather (empty if OPENWEATHER_KEY unset):", scraper.fetch_weather(31.23, 121.47))
