"""Live LLM sentiment stream (treatment 5's perception module).

Open-source fintech model trained on Bitcoin data: ElKulako/cryptobert
(BERTweet pre-trained on 3.2M crypto posts, fine-tuned bullish/bearish).
The FinRL-Contest pattern: the LLM perceives (headline -> signal), the
reinforcement learner decides — LLM weights stay frozen; the RL policy
learns from betting P&L.

Headlines come from open RSS feeds (CoinDesk, Cointelegraph, Decrypt);
each headline is scored once and cached on disk.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import requests

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CACHE = RESULTS_DIR / "headline_scores.json"

FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]
RELEVANT = re.compile(r"bitcoin|btc|crypto|coinbase|etf|satoshi|halving",
                      re.IGNORECASE)

_clf = None  # lazy singleton — the model loads once per process


def _classifier():
    global _clf
    if _clf is None:
        from transformers import pipeline
        _clf = pipeline("text-classification", model="ElKulako/cryptobert",
                        top_k=None)
    return _clf


def fetch_headlines(limit: int = 30) -> list[str]:
    seen, out = set(), []
    for url in FEEDS:
        try:
            xml = requests.get(url, timeout=10,
                               headers={"User-Agent": "btc-rl-poc/0.1"}).text
        except Exception:
            continue
        for title in re.findall(r"<title>(?:<!\[CDATA\[)?([^<\]]{15,200})", xml):
            t = title.strip()
            if RELEVANT.search(t) and t not in seen:
                seen.add(t)
                out.append(t)
    return out[:limit]


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def score_headlines(headlines: list[str]) -> list[float]:
    """Per-headline score in [-1, +1]: P(bullish) - P(bearish). Cached."""
    cache = _load_cache()
    fresh = [h for h in headlines
             if hashlib.sha1(h.encode()).hexdigest()[:16] not in cache]
    if fresh:
        for h, result in zip(fresh, _classifier()(fresh)):
            probs = {r["label"].lower(): r["score"] for r in result}
            cache[hashlib.sha1(h.encode()).hexdigest()[:16]] = round(
                probs.get("bullish", 0) - probs.get("bearish", 0), 4)
        items = dict(list(cache.items())[-2000:])
        CACHE.write_text(json.dumps(items))
        cache = items
    return [cache[hashlib.sha1(h.encode()).hexdigest()[:16]]
            for h in headlines
            if hashlib.sha1(h.encode()).hexdigest()[:16] in cache]


def sentiment_snapshot() -> dict:
    """Current LLM view of the tape: mean sentiment + headline intensity."""
    heads = fetch_headlines()
    scores = score_headlines(heads)
    return {"sent": (sum(scores) / len(scores)) if scores else None,
            "news_n": len(heads)}
