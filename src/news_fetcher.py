"""
News Fetcher for Black-Litterman + LLM Fusion
Handles live news (Finnhub API) and historical news (JSON database)
Also computes simple sentiment scores for dynamic Omega calculation
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentiment helpers (VADER-based, no API key required)
# ---------------------------------------------------------------------------

def _get_sentiment_analyzer():
    """Lazy-load VADER SentimentIntensityAnalyzer."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning("vaderSentiment not installed – sentiment scoring disabled")
        return None

_VADER = None  # cached instance


def score_sentiment(text: str) -> float:
    """
    Score a piece of text with VADER.

    Returns compound score in [-1, 1]:
      > +0.05  → positive
      < -0.05  → negative
      else      → neutral
    """
    global _VADER
    if _VADER is None:
        _VADER = _get_sentiment_analyzer()
    if _VADER is None:
        return 0.0
    return _VADER.polarity_scores(str(text))["compound"]


def sentiment_label(score: float) -> str:
    """Convert VADER compound score to label."""
    if score >= 0.05:
        return "POSITIVE"
    elif score <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# NewsDatabase  – load / save the JSON store used by the backtester
# ---------------------------------------------------------------------------

class NewsDatabase:
    """
    Manages a local JSON file that stores news per {date → ticker → [articles]}.
    Used during backtesting to avoid look-ahead bias.

    Schema
    ------
    {
        "2025-02-28": {
            "AAPL": [
                {
                    "datetime":  <unix timestamp>,
                    "headline":  "...",
                    "summary":   "...",
                    "source":    "SeekingAlpha",
                    "category":  "company",
                    "url":       "...",
                    "sentiment": 0.32          # added by this class
                },
                ...
            ],
            ...
        },
        ...
    }
    """

    DEFAULT_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "news_database.json"
    )

    def __init__(self, path: Optional[str] = None):
        self.path = path or self.DEFAULT_PATH
        self._db: Dict = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._db = json.load(f)
                logger.info(
                    f"NewsDatabase loaded – {len(self._db)} dates  ({self.path})"
                )
            except Exception as e:
                logger.warning(f"Failed to load news database: {e}")
                self._db = {}
        else:
            logger.info("No news database found – starting empty")
            self._db = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._db, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"NewsDatabase saved → {self.path}")

    # ------------------------------------------------------------------
    def get(self, ticker: str, date: pd.Timestamp) -> List[Dict]:
        """
        Retrieve articles for *ticker* published in the 7-day window
        ending on *date* (no look-ahead).
        """
        if not self._db:
            return []

        start = date - timedelta(days=7)

        articles = []
        for date_str, ticker_map in self._db.items():
            try:
                db_date = pd.Timestamp(date_str)
            except Exception:
                continue
            if start < db_date <= date:
                articles.extend(ticker_map.get(ticker, []))

        # Sort by datetime descending, keep top 5
        articles.sort(key=lambda a: a.get("datetime", 0), reverse=True)
        return articles[:5]

    def put(self, date_str: str, ticker: str, articles: List[Dict]):
        """Insert articles into the database."""
        if date_str not in self._db:
            self._db[date_str] = {}
        self._db[date_str][ticker] = articles

    def has_date(self, date_str: str) -> bool:
        return date_str in self._db

    @property
    def dates(self) -> List[str]:
        return sorted(self._db.keys())


# ---------------------------------------------------------------------------
# FinnhubNewsFetcher  – live news from Finnhub API
# ---------------------------------------------------------------------------

class FinnhubNewsFetcher:
    """
    Fetches company news from Finnhub.io.

    API key: set env var FINNHUB_API_KEY or pass explicitly.
    Free tier: 60 calls/minute.

    Usage
    -----
    fetcher = FinnhubNewsFetcher()
    articles = fetcher.fetch(ticker="AAPL", lookback_days=7)

    # Populate news_database for a list of month-end dates
    fetcher.populate_database(db, tickers=MAG7, dates=["2025-02-28", ...])
    """

    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "FINNHUB_API_KEY not set – news fetching will be unavailable"
            )

    # ------------------------------------------------------------------
    def fetch(
        self,
        ticker: str,
        date: Optional[pd.Timestamp] = None,
        lookback_days: int = 7,
        max_articles: int = 5,
    ) -> List[Dict]:
        """
        Fetch recent news for *ticker*.

        Parameters
        ----------
        ticker : str
        date : Timestamp, optional  – end date (defaults to today)
        lookback_days : int
        max_articles : int

        Returns
        -------
        List of dicts with keys: datetime, headline, summary, source, url, sentiment
        """
        if not self.api_key:
            return []

        end_dt = date or pd.Timestamp.now()
        start_dt = end_dt - timedelta(days=lookback_days)

        params = {
            "symbol": ticker,
            "from": start_dt.strftime("%Y-%m-%d"),
            "to": end_dt.strftime("%Y-%m-%d"),
            "token": self.api_key,
        }

        try:
            import requests as req
            resp = req.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            logger.error(f"Finnhub fetch error for {ticker}: {e}")
            return []

        articles = []
        for item in raw[:max_articles]:
            sentiment = score_sentiment(
                f"{item.get('headline', '')} {item.get('summary', '')}"
            )
            articles.append(
                {
                    "datetime": item.get("datetime", 0),
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "category": item.get("category", "company"),
                    "url": item.get("url", ""),
                    "sentiment": sentiment,
                }
            )

        logger.info(f"Fetched {len(articles)} articles for {ticker}")
        return articles

    # ------------------------------------------------------------------
    def populate_database(
        self,
        db: NewsDatabase,
        tickers: List[str],
        dates: List[str],
        lookback_days: int = 7,
        sleep_sec: float = 1.2,   # stay under 60 req/min
    ):
        """
        Populate *db* with news for each (date, ticker) combination.
        Skips dates that already exist in the database.
        """
        for date_str in dates:
            if db.has_date(date_str):
                logger.info(f"Skipping {date_str} – already in database")
                continue
            ts = pd.Timestamp(date_str)
            for ticker in tickers:
                articles = self.fetch(ticker, date=ts, lookback_days=lookback_days)
                db.put(date_str, ticker, articles)
                time.sleep(sleep_sec)

        db.save()
        logger.info("Database population complete")


# ---------------------------------------------------------------------------
# Aggregate sentiment helpers used by llm_view_generator
# ---------------------------------------------------------------------------

def aggregate_sentiment(articles: List[Dict]) -> Dict:
    """
    Given a list of article dicts (with optional 'sentiment' field),
    compute aggregate scores.

    Returns
    -------
    dict with keys:
        score       float  mean VADER compound, or re-scored from text
        label       str    POSITIVE / NEUTRAL / NEGATIVE
        n_articles  int
    """
    if not articles:
        return {"score": 0.0, "label": "NEUTRAL", "n_articles": 0}

    scores = []
    for a in articles:
        if "sentiment" in a and a["sentiment"] is not None:
            scores.append(float(a["sentiment"]))
        else:
            text = f"{a.get('headline', '')} {a.get('summary', '')}"
            scores.append(score_sentiment(text))

    mean_score = float(np.mean(scores))
    return {
        "score": mean_score,
        "label": sentiment_label(mean_score),
        "n_articles": len(articles),
    }


def tech_signal_to_int(alpha_trend: str) -> int:
    """
    Map Alpha Trend classification to [-1, 0, 1] for fusion math.
    ACCELERATING → +1, DECELERATING → -1, STABLE → 0
    """
    mapping = {"ACCELERATING": 1, "STABLE": 0, "DECELERATING": -1}
    return mapping.get(alpha_trend.upper(), 0)


def news_sentiment_to_int(label: str) -> int:
    """Map sentiment label to [-1, 0, 1]."""
    mapping = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
    return mapping.get(label.upper(), 0)


def compute_agreement_score(alpha_trend: str, news_label: str) -> float:
    """
    Compute signal agreement in [-1, 1].
      +1 → both bullish or both bearish (fully aligned)
       0 → one neutral, or unrelated signals
      -1 → directly conflicting

    Used to scale Omega:
        omega_multiplier = base × exp(-k × agreement)
    where k controls sensitivity (default 0.5).
    """
    tech = tech_signal_to_int(alpha_trend)
    news = news_sentiment_to_int(news_label)
    return float(tech * news)  # in {-1, 0, +1}


if __name__ == "__main__":
    # Quick smoke test
    import os
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if api_key:
        fetcher = FinnhubNewsFetcher(api_key)
        articles = fetcher.fetch("AAPL", lookback_days=7)
        for a in articles:
            print(f"[{a['source']}] {a['headline'][:80]}  sentiment={a['sentiment']:.2f}")
    else:
        print("Set FINNHUB_API_KEY to test live fetching")

    db = NewsDatabase()
    articles = db.get("AAPL", pd.Timestamp("2025-03-01"))
    print(f"\nDB lookup AAPL 2025-03-01: {len(articles)} articles")
    if articles:
        agg = aggregate_sentiment(articles)
        print(f"Aggregate sentiment: {agg}")
