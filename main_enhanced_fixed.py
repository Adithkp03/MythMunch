import os
import re
import asyncio
import logging
import hashlib
import time
import json
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter

import requests
import faiss
import numpy as np
import aiohttp
import urllib.parse
import aioredis
import tweepy
import asyncpraw  # CHANGED: Using asyncpraw instead of praw
import networkx as nx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from rank_bm25 import BM25Okapi
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURATION FROM ENVIRONMENT ---
def get_env_bool(key: str, default: bool = False) -> bool:
    """Convert environment variable to boolean"""
    return os.getenv(key, str(default)).lower() in ('true', '1', 'yes', 'on')

def get_env_int(key: str, default: int) -> int:
    """Convert environment variable to integer"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def get_env_float(key: str, default: float) -> float:
    """Convert environment variable to float"""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

# Core API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

# Social Media API Keys (NEW)
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")

# Infrastructure (NEW)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = get_env_int("PORT", 8000)
WORKERS = get_env_int("WORKERS", 1)
RELOAD = get_env_bool("RELOAD", True)

# Application Settings
MAX_WORKERS = get_env_int("MAX_WORKERS", min(64, (os.cpu_count() or 4) * 4))
DEFAULT_CONCURRENCY_LIMIT = get_env_int("DEFAULT_CONCURRENCY_LIMIT", 16)
SIMILARITY_THRESHOLD = get_env_float("SIMILARITY_THRESHOLD", 0.75)
BM25_TOP_K = get_env_int("BM25_TOP_K", 15)
DEFAULT_PASS1_K = get_env_int("DEFAULT_PASS1_K", 25)
DEFAULT_PASS2_K = get_env_int("DEFAULT_PASS2_K", 15)
MAX_OUTPUT_TOKENS = get_env_int("MAX_OUTPUT_TOKENS", 400)

# Caching & Retry
CACHE_TTL = get_env_int("CACHE_TTL", 3600)
RETRY_LIMIT = get_env_int("RETRY_LIMIT", 3)
RETRY_BACKOFF = get_env_int("RETRY_BACKOFF", 2)

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = get_env_bool("DEBUG", True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# AI Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
TEMPERATURE = get_env_float("TEMPERATURE", 0.1)
TOP_P = get_env_float("TOP_P", 0.8)

# Evidence Sources
ENABLE_WIKIPEDIA = get_env_bool("ENABLE_WIKIPEDIA", True)
ENABLE_PUBMED = get_env_bool("ENABLE_PUBMED", True)
ENABLE_NEWSAPI = get_env_bool("ENABLE_NEWSAPI", True)
ENABLE_GOOGLE_SEARCH = get_env_bool("ENABLE_GOOGLE_SEARCH", True)

# NEW: Crisis Features
ENABLE_REAL_TIME_MONITORING = get_env_bool("ENABLE_REAL_TIME_MONITORING", True)
ENABLE_TREND_DETECTION = get_env_bool("ENABLE_TREND_DETECTION", True)
ENABLE_CRISIS_CONTEXT = get_env_bool("ENABLE_CRISIS_CONTEXT", True)
ENABLE_SOCIAL_MEDIA = get_env_bool("ENABLE_SOCIAL_MEDIA", True)

# Source Weights
WIKIPEDIA_WEIGHT = get_env_float("WIKIPEDIA_WEIGHT", 0.85)
PUBMED_WEIGHT = get_env_float("PUBMED_WEIGHT", 0.95)
NEWSAPI_WEIGHT = get_env_float("NEWSAPI_WEIGHT", 0.7)
GOOGLE_SEARCH_WEIGHT = get_env_float("GOOGLE_SEARCH_WEIGHT", 0.7)
GEMINI_WEIGHT = get_env_float("GEMINI_WEIGHT", 1.0)
TWITTER_WEIGHT = get_env_float("TWITTER_WEIGHT", 0.6) # NEW
REDDIT_WEIGHT = get_env_float("REDDIT_WEIGHT", 0.65) # NEW

# Validate required API keys
required_keys = {
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "NEWSAPI_KEY": NEWSAPI_KEY,
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "GOOGLE_CX": GOOGLE_CX
}

missing_keys = [key for key, value in required_keys.items() if not value]
if missing_keys:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")

# Initialize AI models and thread executor
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(LLM_MODEL)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("crisis_detector")

# FastAPI app configuration
app = FastAPI(
    title="Crisis Misinformation Detection API",
    description="AI-powered real-time crisis misinformation detection system with multi-source evidence analysis",
    version="v7.0",
    debug=DEBUG
)

# CORS configuration
try:
    cors_origins = eval(os.getenv("CORS_ORIGINS", '["*"]'))
    cors_methods = eval(os.getenv("CORS_METHODS", '["*"]'))
    cors_headers = eval(os.getenv("CORS_HEADERS", '["*"]'))
except:
    cors_origins = ["*"]
    cors_methods = ["*"]
    cors_headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=cors_methods,
    allow_headers=cors_headers,
    allow_credentials=True,
)

# Source credibility weights
BASE_SOURCE_CREDIBILITY = {
    "gemini": GEMINI_WEIGHT,
    "wikipedia": WIKIPEDIA_WEIGHT,
    "pubmed": PUBMED_WEIGHT,
    "newsapi": NEWSAPI_WEIGHT,
    "google_search": GOOGLE_SEARCH_WEIGHT,
    "twitter": TWITTER_WEIGHT,
    "reddit": REDDIT_WEIGHT,
}

dynamic_source_credibility = BASE_SOURCE_CREDIBILITY.copy()

# --- CRISIS DATABASE CLASS ---
class CrisisDatabase:
    def __init__(self):
        self.active_crises = {
            "ukraine_war": {
                "name": "Ukraine-Russia Conflict",
                "keywords": ["ukraine", "russia", "putin", "zelensky", "kyiv", "moscow", "war", "invasion", "donbas", "crimea"],
                "entities": ["vladimir putin", "volodymyr zelensky", "ukraine", "russia", "nato", "eu"],
                "start_date": "2022-02-24",
                "status": "ongoing",
                "severity": "high",
                "severity_score": 0.95
            },
            "covid_pandemic": {
                "name": "COVID-19 Pandemic",
                "keywords": ["covid", "coronavirus", "vaccine", "pandemic", "lockdown", "mask", "omicron", "delta"],
                "entities": ["who", "cdc", "pfizer", "moderna", "fauci", "china", "wuhan"],
                "start_date": "2020-01-01",
                "status": "ongoing",
                "severity": "high",
                "severity_score": 0.8
            },
            "climate_crisis": {
                "name": "Climate Change Crisis",
                "keywords": ["climate change", "global warming", "carbon emissions", "renewable energy", "paris agreement", "cop28"],
                "entities": ["ipcc", "un", "greta thunberg", "cop28", "paris agreement"],
                "start_date": "2000-01-01",
                "status": "ongoing",
                "severity": "high",
                "severity_score": 0.9
            },
            "us_election_2024": {
                "name": "US Presidential Election 2024",
                "keywords": ["election", "voting", "ballot", "fraud", "rigged", "stolen", "trump", "biden"],
                "entities": ["donald trump", "joe biden", "election", "voting", "ballot"],
                "start_date": "2023-01-01",
                "status": "ongoing",
                "severity": "medium",
                "severity_score": 0.7
            }
        }

    def get_relevant_crises(self, claim: str) -> List[Dict]:
        """Find crises relevant to a claim"""
        claim_lower = claim.lower()
        relevant = []
        
        for crisis_id, crisis_data in self.active_crises.items():
            score = 0
            matched_keywords = []
            
            for keyword in crisis_data["keywords"]:
                if keyword in claim_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            for entity in crisis_data["entities"]:
                if entity.lower() in claim_lower:
                    score += 2
                    matched_keywords.append(entity)
            
            if score > 0:
                crisis_data["relevance_score"] = score
                crisis_data["crisis_id"] = crisis_id
                crisis_data["keywords_matched"] = matched_keywords
                relevant.append(crisis_data.copy())
        
        return sorted(relevant, key=lambda x: x["relevance_score"], reverse=True)

# --- SOCIAL MEDIA MONITOR CLASS ---
class SocialMediaMonitor:
    def __init__(self):
        self.twitter_client = None
        self.reddit_api = None
        self.last_twitter_call = 0
        self.last_reddit_call = 0

    async def setup_apis(self):
        """Setup social media API clients"""
        try:
            # Twitter API v2 setup
            if TWITTER_BEARER_TOKEN:
                self.twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
                logger.info("Twitter API initialized")
            
            # Reddit API setup with ASYNC PRAW
            if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
                self.reddit_api = asyncpraw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent="CrisisMisinfoDetector/1.0"
                )
                logger.info("Async Reddit API initialized")
                
        except Exception as e:
            logger.error(f"Social media API setup error: {e}")

    async def search_twitter(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search recent tweets with rate limiting"""
        if not self.twitter_client or not ENABLE_SOCIAL_MEDIA:
            return []
        
        # Rate limiting check
        now = time.time()
        if now - self.last_twitter_call < 5:  # 5 seconds minimum between calls
            logger.info("Twitter rate limit: waiting...")
            await asyncio.sleep(5)
        
        try:
            self.last_twitter_call = time.time()
            await asyncio.sleep(3)  # Additional safety delay
            
            def _search():
                try:
                    return self.twitter_client.search_recent_tweets(
                        query=query,
                        max_results=min(max_results, 10),  # Reduced to 10
                        tweet_fields=['created_at', 'public_metrics', 'author_id']
                    )
                except Exception as e:
                    logger.warning(f"Twitter API error: {e}")
                    return None
            
            # Run in executor to avoid blocking
            tweets_response = await asyncio.get_event_loop().run_in_executor(None, _search)
            
            if not tweets_response or not tweets_response.data:
                return []
            
            results = []
            for tweet in tweets_response.data:
                results.append({
                    "source": "twitter",
                    "title": f"Tweet by user {tweet.author_id}",
                    "url": f"https://twitter.com/user/status/{tweet.id}",
                    "snippet": tweet.text[:500],
                    "credibility": dynamic_source_credibility.get("twitter", 0.6),
                    "timestamp": tweet.created_at.isoformat() if tweet.created_at else datetime.utcnow().isoformat(),
                    "engagement": {
                        "retweets": tweet.public_metrics.get('retweet_count', 0) if tweet.public_metrics else 0,
                        "likes": tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                        "replies": tweet.public_metrics.get('reply_count', 0) if tweet.public_metrics else 0
                    },
                    "platform_specific": {
                        "tweet_id": str(tweet.id),
                        "author_id": str(tweet.author_id),
                        "public_metrics": tweet.public_metrics if tweet.public_metrics else {}
                    }
                })
            return results
            
        except Exception as e:
            logger.warning(f"Twitter search error: {e}")
            return []

    async def search_reddit(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Reddit posts with async PRAW"""
        if not self.reddit_api or not ENABLE_SOCIAL_MEDIA:
            return []
        
        # Rate limiting check
        now = time.time()
        if now - self.last_reddit_call < 3:  # 3 seconds minimum between calls
            logger.info("Reddit rate limit: waiting...")
            await asyncio.sleep(3)
        
        try:
            self.last_reddit_call = time.time()
            results = []
            subreddit = await self.reddit_api.subreddit("all")
            
            async for submission in subreddit.search(query, limit=max_results):
                results.append({
                    "source": "reddit",
                    "title": submission.title[:200],
                    "url": f"https://reddit.com{submission.permalink}",
                    "snippet": (submission.selftext or submission.title)[:500],
                    "credibility": dynamic_source_credibility.get("reddit", 0.65),
                    "timestamp": datetime.fromtimestamp(submission.created_utc).isoformat(),
                    "engagement": {
                        "upvotes": submission.score,
                        "comments": submission.num_comments,
                        "upvote_ratio": getattr(submission, 'upvote_ratio', 0)
                    },
                    "platform_specific": {
                        "subreddit": submission.subreddit.display_name,
                        "post_id": submission.id,
                        "is_self": submission.is_self
                    }
                })
            return results
            
        except Exception as e:
            logger.warning(f"Reddit search error: {e}")
            return []

    async def close(self):
        """Close async connections"""
        if self.reddit_api:
            await self.reddit_api.close()

# --- STREAM PROCESSOR CLASS ---
class MisinformationStreamProcessor:
    def __init__(self):
        self.redis_client = None
        self.active_streams = {}
        self.trend_buffer = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitor_task = None

    async def setup_redis(self):
        """Setup Redis connection for stream processing"""
        try:
            self.redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None

    async def start_monitoring_streams(self, keywords: List[str]):
        """Start monitoring social media streams for keywords with single instance control"""
        if not keywords or not ENABLE_REAL_TIME_MONITORING:
            return

        # Prevent multiple instances
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        logger.info(f"Starting stream monitoring for keywords: {keywords}")

        # Create single monitoring task for all keywords
        try:
            self.monitor_task = asyncio.create_task(self.monitor_all_keywords(keywords))
            await self.monitor_task
        except Exception as e:
            logger.error(f"Stream monitoring error: {e}")
        finally:
            self.monitoring_active = False

    async def monitor_all_keywords(self, keywords: List[str]):
        """Monitor all keywords in single loop to avoid rate limits"""
        social_monitor = SocialMediaMonitor()
        await social_monitor.setup_apis()
        
        try:
            cycle = 0
            while self.monitoring_active:
                cycle += 1
                logger.info(f"Monitoring cycle {cycle} starting...")
                
                for i, keyword in enumerate(keywords):
                    if not self.monitoring_active:
                        break
                    
                    try:
                        # Stagger requests - 30 seconds between keywords
                        if i > 0:
                            await asyncio.sleep(30)
                        
                        logger.info(f"Checking keyword: {keyword}")
                        
                        twitter_mentions = await social_monitor.search_twitter(keyword, 5)
                        reddit_mentions = await social_monitor.search_reddit(keyword, 3)
                        
                        all_mentions = twitter_mentions + reddit_mentions
                        logger.info(f"Keyword '{keyword}': {len(all_mentions)} mentions found")
                        
                        for mention in all_mentions:
                            # Add to trend buffer
                            buffer_item = {
                                "keyword": keyword,
                                "mention": mention,
                                "timestamp": datetime.now().isoformat(),
                                "sentiment": await self.analyze_sentiment(mention["snippet"])
                            }
                            self.trend_buffer.append(buffer_item)
                            
                            # Check for suspicious patterns
                            if await self.detect_suspicious_pattern(mention):
                                await self.trigger_misinformation_alert(keyword, mention)
                                
                    except Exception as e:
                        logger.error(f"Error monitoring {keyword}: {e}")
                        await asyncio.sleep(60)  # Extra delay on error
                
                # Wait 10 minutes before next cycle
                logger.info("Monitoring cycle complete, waiting 10 minutes...")
                await asyncio.sleep(600)  # 10 minutes
                
        except asyncio.CancelledError:
            logger.info("Monitoring cancelled")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            await social_monitor.close()

    async def stop_monitoring(self):
        """Stop monitoring"""
        if self.monitor_task and not self.monitor_task.done():
            self.monitoring_active = False
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Monitoring stopped")

    async def analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (could be enhanced with proper NLP)"""
        negative_words = ["fake", "false", "hoax", "lie", "scam", "fraud", "conspiracy", "corrupt"]
        positive_words = ["true", "fact", "verified", "confirmed", "legitimate", "authentic"]
        
        text_lower = text.lower()
        neg_score = sum(1 for word in negative_words if word in text_lower)
        pos_score = sum(1 for word in positive_words if word in text_lower)
        
        if neg_score + pos_score == 0:
            return 0.0
        
        return (pos_score - neg_score) / (pos_score + neg_score)

    async def detect_suspicious_pattern(self, mention: Dict) -> bool:
        """Detect if a mention shows suspicious misinformation patterns"""
        suspicious_indicators = 0
        content = mention.get("snippet", "").lower()

        # Pattern 1: Urgent language
        urgent_words = ["breaking", "urgent", "immediately", "share now", "before they delete", "act fast"]
        if any(word in content for word in urgent_words):
            suspicious_indicators += 1

        # Pattern 2: Conspiracy language
        conspiracy_words = ["they don't want you to know", "mainstream media hiding", "cover up", "wake up sheeple"]
        if any(phrase in content for phrase in conspiracy_words):
            suspicious_indicators += 2

        # Pattern 3: Unverified claims
        unverified_words = ["sources say", "reportedly", "allegedly", "rumor has it", "i heard"]
        if any(word in content for word in unverified_words):
            suspicious_indicators += 1

        # Pattern 4: High engagement on suspicious content
        engagement = mention.get("engagement", {})
        total_engagement = sum(engagement.values()) if engagement else 0
        if total_engagement > 500 and suspicious_indicators > 0:
            suspicious_indicators += 2

        return suspicious_indicators >= 3

    async def trigger_misinformation_alert(self, keyword: str, mention: Dict):
        """Trigger alert for potential misinformation"""
        alert_id = f"alert_{int(datetime.now().timestamp())}"
        alert = {
            "alert_id": alert_id,
            "keyword": keyword,
            "mention": mention,
            "timestamp": datetime.now().isoformat(),
            "severity": "high",
            "status": "active",
            "confidence": 0.8
        }

        # Store alert in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"alert:{alert_id}",
                    7200,  # 2 hours expiry
                    json.dumps(alert)
                )
            except Exception as e:
                logger.error(f"Failed to store alert in Redis: {e}")

        # Log alert
        logger.warning(f"MISINFORMATION ALERT: {keyword} - {mention.get('snippet', '')[:100]}...")
        return alert

    async def get_trend_analysis(self, time_window_minutes: int = 60) -> Dict:
        """Analyze trends from recent buffer"""
        cutoff_time = datetime.now().timestamp() - (time_window_minutes * 60)
        recent_items = [
            item for item in self.trend_buffer
            if datetime.fromisoformat(item["timestamp"]).timestamp() > cutoff_time
        ]

        # Count mentions per keyword
        keyword_counts = {}
        sentiment_scores = {}
        for item in recent_items:
            keyword = item["keyword"]
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            if keyword not in sentiment_scores:
                sentiment_scores[keyword] = []
            sentiment_scores[keyword].append(item["sentiment"])

        # Calculate trending keywords
        trending = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "trending_keywords": trending[:10],
            "total_mentions": len(recent_items),
            "time_window_minutes": time_window_minutes,
            "sentiment_analysis": {
                k: sum(scores) / len(scores) for k, scores in sentiment_scores.items()
            },
            "analysis_timestamp": datetime.now().isoformat()
        }

    async def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts from Redis"""
        if not self.redis_client:
            return []

        try:
            alert_keys = await self.redis_client.keys("alert:*")
            alerts = []
            for key in alert_keys:
                alert_data = await self.redis_client.get(key)
                if alert_data:
                    alerts.append(json.loads(alert_data))
            return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            logger.error(f"Failed to get alerts from Redis: {e}")
            return []

# --- TREND DETECTOR CLASS ---
class TrendDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.recent_content = []
        self.content_timestamps = []

    def add_content(self, content: str, timestamp: datetime = None):
        """Add content for trend analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.recent_content.append(content)
        self.content_timestamps.append(timestamp)
        
        # Keep only last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        valid_indices = [i for i, ts in enumerate(self.content_timestamps) if ts > cutoff]
        self.recent_content = [self.recent_content[i] for i in valid_indices]
        self.content_timestamps = [self.content_timestamps[i] for i in valid_indices]

    def detect_emerging_clusters(self, min_samples: int = 3) -> List[Dict]:
        """Detect clusters of similar content that might indicate coordinated misinformation"""
        if len(self.recent_content) < min_samples or not ENABLE_TREND_DETECTION:
            return []

        try:
            # Vectorize content
            content_vectors = self.vectorizer.fit_transform(self.recent_content)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=0.3, min_samples=min_samples, metric='cosine')
            cluster_labels = clustering.fit_predict(content_vectors.toarray())
            
            # Analyze clusters
            clusters = []
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_content = [self.recent_content[i] for i in cluster_indices]
                cluster_timestamps = [self.content_timestamps[i] for i in cluster_indices]
                
                # Calculate cluster metrics
                time_span = max(cluster_timestamps) - min(cluster_timestamps)
                velocity = len(cluster_content) / max(time_span.total_seconds() / 3600, 1)  # posts per hour
                
                # Extract common terms
                cluster_text = " ".join(cluster_content)
                cluster_vector = self.vectorizer.transform([cluster_text])
                feature_names = self.vectorizer.get_feature_names_out()
                scores = cluster_vector.toarray()[0]
                top_terms = [(feature_names[i], scores[i]) for i in scores.argsort()[-10:][::-1]]
                
                clusters.append({
                    "cluster_id": cluster_id,
                    "size": len(cluster_content),
                    "velocity": velocity,
                    "time_span_hours": time_span.total_seconds() / 3600,
                    "top_terms": top_terms,
                    "sample_content": cluster_content[:3],
                    "risk_score": self.calculate_risk_score(len(cluster_content), velocity, top_terms),
                    "first_seen": min(cluster_timestamps).isoformat(),
                    "last_seen": max(cluster_timestamps).isoformat()
                })
            
            return sorted(clusters, key=lambda x: x["risk_score"], reverse=True)
        except Exception as e:
            logger.error(f"Cluster detection error: {e}")
            return []

    def calculate_risk_score(self, size: int, velocity: float, top_terms: List) -> float:
        """Calculate risk score for a content cluster"""
        risk_score = 0.0
        
        # Size factor (more posts = higher risk)
        risk_score += min(size / 10, 1.0) * 0.3
        
        # Velocity factor (rapid spread = higher risk)
        risk_score += min(velocity / 5, 1.0) * 0.4
        
        # Content analysis (suspicious terms = higher risk)
        suspicious_terms = [
            "fake", "hoax", "conspiracy", "cover-up", "they don't want you to know",
            "mainstream media lies", "wake up", "do your research", "before it's deleted"
        ]
        
        term_text = " ".join([term[0] for term in top_terms]).lower()
        suspicious_count = sum(1 for sus_term in suspicious_terms if sus_term in term_text)
        risk_score += min(suspicious_count / 5, 1.0) * 0.3
        
        return min(risk_score, 1.0)

# --- DEMO SCENARIOS CLASS ---
class DemoScenarios:
    def __init__(self):
        self.scenarios = {
            "ukraine_deepfake": {
                "name": "Ukraine Deepfake Video",
                "claim": "Video shows Ukrainian president Zelensky surrendering to Russia",
                "actual_status": "DEEPFAKE - Video uses AI face swap technology",
                "crisis_context": "ukraine_war",
                "demo_evidence": [
                    {
                        "source": "twitter",
                        "title": "Verified fact-checker debunks deepfake video",
                        "snippet": "BREAKING: Zelensky surrender video is FAKE! Digital forensics show clear deepfake artifacts.",
                        "credibility": 0.9,
                        "url": "https://twitter.com/factchecker/status/123456",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "source": "reuters",
                        "title": "Fact Check: Video of Zelensky calling for surrender is a deepfake",
                        "snippet": "Reuters fact-check confirms video is manipulated using AI technology.",
                        "credibility": 0.95,
                        "url": "https://reuters.com/factcheck/deepfake-zelensky",
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "expected_verdict": "FALSE",
                "misinformation_type": "deepfake",
                "potential_harm": "high"
            },
            "vaccine_5g": {
                "name": "COVID Vaccine 5G Conspiracy",
                "claim": "COVID vaccines contain 5G microchips for government tracking",
                "actual_status": "CONSPIRACY THEORY - No scientific evidence",
                "crisis_context": "covid_pandemic",
                "demo_evidence": [
                    {
                        "source": "pubmed",
                        "title": "Comprehensive analysis of mRNA vaccine ingredients",
                        "snippet": "Comprehensive analysis of mRNA vaccine ingredients shows no metallic components or microchips",
                        "credibility": 0.98,
                        "url": "https://pubmed.ncbi.nlm.nih.gov/123456",
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "expected_verdict": "FALSE",
                "misinformation_type": "conspiracy_theory",
                "potential_harm": "medium"
            }
        }

    def run_demo_scenario(self, scenario_name: str) -> Dict:
        """Run a demo scenario and return simulated results"""
        if scenario_name not in self.scenarios:
            return {"error": "Scenario not found"}
        
        scenario = self.scenarios[scenario_name]
        
        # Simulate the fact-checking process
        return {
            "scenario_name": scenario_name,
            "claim": scenario["claim"],
            "verdict": scenario["expected_verdict"],
            "confidence": 0.92,
            "explanation": f"Analysis shows this claim is {scenario['actual_status']}",
            "evidence": scenario["demo_evidence"],
            "crisis_context": [{"crisis_id": scenario["crisis_context"], "name": scenario.get("crisis_context", "")}],
            "misinformation_type": scenario["misinformation_type"],
            "potential_harm": scenario["potential_harm"],
            "detection_time": "Real-time (< 2 minutes)",
            "intervention_recommended": True,
            "demo_timestamp": datetime.now().isoformat()
        }

    def get_all_scenarios(self) -> Dict:
        """Get all available demo scenarios"""
        return {
            name: {
                "name": scenario["name"],
                "claim": scenario["claim"],
                "crisis_context": scenario["crisis_context"],
                "misinformation_type": scenario["misinformation_type"],
                "potential_harm": scenario["potential_harm"]
            }
            for name, scenario in self.scenarios.items()
        }

# --- PYDANTIC MODELS ---
class FactCheckRequest(BaseModel):
    claim: str
    include_evidence: bool = True
    max_evidence_sources: Optional[int] = 8
    detailed_analysis: bool = False

class CrisisFactCheckRequest(BaseModel):
    claim: str
    include_evidence: bool = True
    max_evidence_sources: Optional[int] = 12
    detailed_analysis: bool = False
    monitor_social_media: bool = True
    crisis_context: bool = True

class EvidenceItem(BaseModel):
    source: str
    title: str
    url: str
    snippet: str
    credibility: float
    similarity: float
    timestamp: str

class CrisisContext(BaseModel):
    crisis_id: str
    crisis_name: str
    relevance_score: int
    keywords_matched: List[str]

class FactCheckResponse(BaseModel):
    analysis_id: str
    claim: str
    verdict: str
    confidence: float
    explanation: str
    evidence: List[EvidenceItem]
    gemini_verdict: str
    evidence_consensus: Dict[str, float]
    processing_time: float
    final_verdict_display: str

class EnhancedFactCheckResponse(BaseModel):
    analysis_id: str
    claim: str
    verdict: str
    confidence: float
    explanation: str
    evidence: List[EvidenceItem]
    gemini_verdict: str
    evidence_consensus: Dict[str, float]
    processing_time: float
    final_verdict_display: str
    crisis_context: List[CrisisContext]
    social_media_mentions: int
    trend_indicators: Dict[str, float]
    risk_indicators: Dict[str, float]
    intervention_recommendations: List[str]

# --- RETRY DECORATOR FOR RESILIENCE ---
def async_retry(retries=RETRY_LIMIT, backoff=RETRY_BACKOFF):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logger.warning(f"Attempt {attempt + 1}/{retries} failed for {func.__name__}: {e}")
                    await asyncio.sleep(backoff * (2 ** attempt))
            raise last_exc
        return wrapper
    return decorator

# --- EVIDENCE-INFORMED GEMINI FACT-CHECKING ---
@async_retry()
async def gemini_fact_check_with_evidence_override(claim: str, evidence_consensus: Dict[str, float], crisis_context: List[Dict] = None) -> Dict[str, str]:
    """Enhanced fact-checking that considers evidence and crisis context when making decisions"""
    
    # Build evidence summary for the prompt
    evidence_summary = ""
    if evidence_consensus:
        supports = evidence_consensus.get("supports", 0)
        refutes = evidence_consensus.get("refutes", 0)
        neutral = evidence_consensus.get("neutral", 0)
        evidence_summary = f"\n\nEVIDENCE SUMMARY:\n- Supporting evidence: {supports:.1%}\n- Contradicting evidence: {refutes:.1%}\n- Neutral evidence: {neutral:.1%}"

    # Add crisis context
    crisis_summary = ""
    if crisis_context and ENABLE_CRISIS_CONTEXT:
        crisis_names = [c.get("name", c.get("crisis_id", "")) for c in crisis_context]
        crisis_summary = f"\n\nCRISIS CONTEXT: This claim relates to: {', '.join(crisis_names)}"

    prompt = f"""You are an expert crisis misinformation detector. Analyze this claim carefully in the context of current global events:

CLAIM: "{claim}"

{evidence_summary}

{crisis_summary}

Consider the evidence summary and crisis context above when making your determination. If evidence strongly contradicts the claim (>70% refutation), the claim is likely FALSE. If evidence strongly supports the claim (>70% support), it's likely TRUE.

Important guidelines:

- Trust the evidence consensus when it's overwhelming (>70% in either direction)
- Pay special attention to crisis-related misinformation patterns
- Consider current factual accuracy as of August 2025
- Be especially careful with political positions, health claims, and crisis-related information
- Look for signs of coordinated disinformation campaigns

Provide your analysis in this exact format:

VERDICT: [TRUE/FALSE/PARTIALLY_TRUE/INSUFFICIENT_INFO]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [Brief explanation considering evidence, crisis context, and your knowledge]
KEY_POINTS: [Main points supporting your verdict]

Be thorough but concise. Consider:
1. Your factual knowledge about the claim
2. The evidence summary provided above
3. The crisis context and its implications
4. Logical consistency and relationships between entities
5. Potential for harm if misinformation spreads

Claim: {claim}"""

    def _check():
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    top_p=TOP_P
                ),
            )
            
            text = response.text.strip()
            verdict_match = re.search(r"VERDICT:\s*([A-Z_]+)", text)
            confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
            explanation_match = re.search(r"EXPLANATION:\s*(.+?)(?=KEY_POINTS:|$)", text, re.DOTALL)
            
            return {
                "verdict": verdict_match.group(1) if verdict_match else "INSUFFICIENT_INFO",
                "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
                "explanation": explanation_match.group(1).strip() if explanation_match else "Unable to analyze claim",
                "raw_response": text,
            }
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "verdict": "INSUFFICIENT_INFO",
                "confidence": 0.0,
                "explanation": f"Gemini API error: {str(e)}",
                "raw_response": "",
            }
    
    return await asyncio.get_event_loop().run_in_executor(executor, _check)

# --- EVIDENCE RETRIEVAL FUNCTIONS ---
@async_retry()
async def search_wikipedia(query: str, max_results: int = 5) -> List[Dict]:
    if not ENABLE_WIKIPEDIA:
        return []
    
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "format": "json",
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(search_url, params=params) as resp:
            if resp.status != 200:
                logger.warning(f"Wikipedia search failed with status {resp.status}")
                return []
            
            data = await resp.json()
            titles = [item["title"] for item in data.get("query", {}).get("search", [])]
            
            evidence = []
            for title in titles:
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                try:
                    async with session.get(summary_url) as summary_resp:
                        if summary_resp.status == 200:
                            summary_data = await summary_resp.json()
                            evidence.append({
                                "source": "wikipedia",
                                "title": title,
                                "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                                "snippet": summary_data.get("extract", "")[:500],
                                "credibility": dynamic_source_credibility["wikipedia"],
                                "timestamp": datetime.utcnow().isoformat(),
                            })
                except Exception as e:
                    logger.warning(f"Failed to fetch Wikipedia summary for {title}: {e}")
            
            return evidence

@async_retry()
async def search_pubmed(query: str, max_results: int = 5) -> List[Dict]:
    if not ENABLE_PUBMED:
        return []
    
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(search_url, params=params) as resp:
            if resp.status != 200:
                logger.warning(f"PubMed search failed with status {resp.status}")
                return []
            
            data = await resp.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            
            evidence = []
            for pmid in pmids[:max_results]:
                fetch_params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
                try:
                    async with session.get(fetch_url, params=fetch_params) as fetch_resp:
                        if fetch_resp.status != 200:
                            continue
                        
                        xml = await fetch_resp.text()
                        title_match = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", xml, re.DOTALL)
                        abstract_match = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", xml, re.DOTALL)
                        
                        title_text = (
                            re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
                            if title_match
                            else "PubMed Article"
                        )
                        
                        abstract_text = (
                            re.sub(r"<[^>]+>", "", abstract_match.group(1)).strip()
                            if abstract_match
                            else ""
                        )
                        
                        evidence.append({
                            "source": "pubmed",
                            "title": title_text,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            "snippet": abstract_text[:500],
                            "credibility": dynamic_source_credibility["pubmed"],
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                except Exception as e:
                    logger.warning(f"Failed to fetch PubMed article {pmid}: {e}")
            
            return evidence

@async_retry()
async def search_newsapi(query: str, max_results: int = 5) -> List[Dict]:
    if not ENABLE_NEWSAPI or not NEWSAPI_KEY:
        return []
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": max_results,
        "apiKey": NEWSAPI_KEY,
        "sortBy": "relevance",
        "from": datetime.utcnow().replace(day=1).strftime("%Y-%m-%d"),
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.warning(f"NewsAPI failed with status {resp.status}")
                return []
            
            data = await resp.json()
            articles = data.get("articles", [])
            
            evidence = []
            for article in articles:
                evidence.append({
                    "source": "newsapi",
                    "title": article.get("title", "")[:200],
                    "url": article.get("url", ""),
                    "snippet": article.get("description", "")[:500],
                    "credibility": dynamic_source_credibility["newsapi"],
                    "timestamp": article.get("publishedAt", datetime.utcnow().isoformat()),
                })
            
            return evidence

@async_retry()
async def search_google(query: str, max_results: int = 5) -> List[Dict]:
    if not ENABLE_GOOGLE_SEARCH or not GOOGLE_API_KEY or not GOOGLE_CX:
        return []
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "num": max_results,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                logger.warning(f"Google Custom Search failed with status {resp.status}")
                return []
            
            data = await resp.json()
            items = data.get("items", [])
            
            evidence = []
            for item in items:
                evidence.append({
                    "source": "google_search",
                    "title": item.get("title", "")[:200],
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")[:500],
                    "credibility": dynamic_source_credibility["google_search"],
                    "timestamp": datetime.utcnow().isoformat(),
                })
            
            return evidence

# --- ADVANCED RANKING ---
def build_bm25_index(texts: List[str]) -> BM25Okapi:
    tokenized_texts = [re.findall(r"\w+", text.lower()) for text in texts]
    return BM25Okapi(tokenized_texts)

async def rank_evidence_advanced(claim: str, evidence_list: List[Dict]) -> List[Dict]:
    if not evidence_list:
        return []
    
    texts = [f"{ev.get('title', '')} {ev.get('snippet', '')}" for ev in evidence_list]
    bm25 = build_bm25_index(texts)
    tokenized_query = re.findall(r"\w+", claim.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    
    claim_emb = embedding_model.encode([claim])
    text_embeddings = embedding_model.encode(texts)
    semantic_sims = util.cos_sim(claim_emb, text_embeddings)[0]
    
    for i, ev in enumerate(evidence_list):
        bm25_score = float(bm25_scores[i]) if i < len(bm25_scores) else 0.0
        semantic_score = float(semantic_sims[i])
        credibility = ev.get("credibility", 0.5)
        
        combined_score = semantic_score * 0.45 + bm25_score * 0.35 + credibility * 0.2
        ev["similarity"] = semantic_score
        ev["bm25_score"] = bm25_score
        ev["combined_score"] = combined_score
    
    return sorted(evidence_list, key=lambda x: x["combined_score"], reverse=True)

# --- ENHANCED EVIDENCE ANALYSIS ---
def analyze_evidence_consensus_enhanced(evidence_list: List[Dict], claim: str) -> Dict[str, float]:
    """Enhanced evidence analysis with better contradiction detection"""
    if not evidence_list:
        return {"supports": 0.0, "refutes": 0.0, "neutral": 1.0}
    
    supports, refutes, neutral = 0.0, 0.0, 0.0
    claim_lower = claim.lower()
    
    for ev in evidence_list:
        text = (ev.get("title", "") + " " + ev.get("snippet", "")).lower()
        weight = ev.get("credibility", 0.5) * ev.get("similarity", 0.5)
        
        # Enhanced analysis logic
        is_supporting = False
        is_refuting = False
        
        # Strong support indicators
        strong_support = ["is the", "serves as", "currently", "has been", "elected as", "appointed as", "holds the position", "who is the", "confirmed", "verified"]
        moderate_support = ["confirms", "proves", "demonstrates", "shows", "supports", "validates", "since", "as of"]
        
        # Strong refutation indicators
        strong_refute = ["is not", "was never", "has never been", "incorrect", "false", "myth", "debunked", "not the", "fake", "hoax"]
        moderate_refute = ["former", "ended", "resigned", "stepped down", "no longer", "was the", "allegedly", "rumored"]
        
        # Crisis-specific patterns
        if any(word in claim_lower for word in ["covid", "vaccine", "pandemic"]):
            # COVID/vaccine misinformation patterns
            vaccine_misinfo = ["microchip", "5g", "magnetism", "dna alteration", "population control"]
            if any(term in text for term in vaccine_misinfo):
                is_refuting = True
        
        if any(word in claim_lower for word in ["ukraine", "russia", "war", "putin", "zelensky"]):
            # War misinformation patterns
            war_misinfo = ["biolab", "nazi", "staged", "crisis actor", "false flag"]
            if any(term in text for term in war_misinfo):
                is_refuting = True
        
        # General support/refute analysis
        if not is_supporting and not is_refuting:
            strong_support_count = sum(1 for indicator in strong_support if indicator in text)
            moderate_support_count = sum(1 for indicator in moderate_support if indicator in text)
            strong_refute_count = sum(1 for indicator in strong_refute if indicator in text)
            moderate_refute_count = sum(1 for indicator in moderate_refute if indicator in text)
            
            support_score = strong_support_count * 2 + moderate_support_count
            refute_score = strong_refute_count * 2 + moderate_refute_count
            
            if support_score > refute_score and support_score > 0:
                is_supporting = True
            elif refute_score > support_score and refute_score > 0:
                is_refuting = True
        
        # Apply weights
        if is_supporting:
            supports += weight
        elif is_refuting:
            refutes += weight * 1.1  # Slight boost to refutation detection
        else:
            neutral += weight * 0.8  # Reduce neutral weight to amplify clear signals
    
    total = supports + refutes + neutral
    if total == 0:
        return {"supports": 0.0, "refutes": 0.0, "neutral": 1.0}
    
    return {
        "supports": supports / total,
        "refutes": refutes / total,
        "neutral": neutral / total,
    }

# --- UTILITY FUNCTIONS ---
def calculate_viral_potential(evidence: List[Dict]) -> float:
    """Calculate potential for viral spread"""
    social_evidence = [e for e in evidence if e.get("source") in ["twitter", "reddit"]]
    if not social_evidence:
        return 0.0
    
    total_engagement = 0
    for e in social_evidence:
        engagement = e.get("engagement", {})
        if isinstance(engagement, dict):
            total_engagement += sum(engagement.values())
    
    return min(total_engagement / 10000, 1.0)

def calculate_misinformation_risk(evidence: List[Dict], clusters: List[Dict]) -> float:
    """Calculate overall misinformation risk"""
    base_risk = 0.3
    
    # High-risk sources
    suspicious_sources = sum(1 for e in evidence if any(term in e.get("url", "").lower() for term in ["blog", "telegram", "rumble"]))
    if suspicious_sources > 0:
        base_risk += 0.2
    
    # Cluster analysis
    high_risk_clusters = [c for c in clusters if c["risk_score"] > 0.7]
    if high_risk_clusters:
        base_risk += 0.3
    
    # Social media velocity
    social_evidence = [e for e in evidence if e.get("source") in ["twitter", "reddit"]]
    if len(social_evidence) > len(evidence) * 0.5:  # More than 50% social media
        base_risk += 0.2
    
    return min(base_risk, 1.0)

def generate_intervention_recommendations(verdict: str, risk_indicators: Dict, crisis_context: List) -> List[str]:
    """Generate actionable intervention recommendations"""
    recommendations = []
    
    if verdict == "FALSE" and risk_indicators.get("viral_potential", 0) > 0.7:
        recommendations.append("🚨 URGENT: Issue immediate fact-check alert to partner organizations")
        recommendations.append("📱 Recommend platform-specific interventions (warning labels, reduced distribution)")
    
    crisis_severity = max([c.get("severity_score", 0) for c in crisis_context] + [0])
    if crisis_severity > 0.8:
        recommendations.append("🏛️ Alert relevant government crisis communication teams")
        recommendations.append("📰 Coordinate with news organizations for counter-narrative")
    
    if risk_indicators.get("coordination_risk", 0) > 0.6:
        recommendations.append("🕵️ Investigate potential coordinated inauthentic behavior")
        recommendations.append("🔍 Enhanced monitoring of related accounts and content")
    
    if risk_indicators.get("misinformation_risk", 0) > 0.8:
        recommendations.append("⚠️ Implement proactive content moderation measures")
        recommendations.append("📢 Prepare authoritative counter-messaging")
    
    return recommendations

# --- MAIN FACT CHECKER CLASS ---
class EnhancedFactChecker:
    def __init__(self):
        self.cache = {}
        self.crisis_db = CrisisDatabase()
        self.social_monitor = SocialMediaMonitor()

    async def gather_evidence_enhanced(self, claim: str, max_sources: int = 12) -> List[Dict]:
        """Enhanced evidence gathering with social media"""
        logger.info(f"Gathering evidence for claim: {claim[:50]}...")
        
        # Get crisis context
        relevant_crises = self.crisis_db.get_relevant_crises(claim)
        
        # Traditional sources + social media
        tasks = []
        base_sources = 4 if not ENABLE_SOCIAL_MEDIA else 4
        social_sources = 2 if ENABLE_SOCIAL_MEDIA else 0
        total_sources = base_sources + social_sources
        sources_per_type = max_sources // total_sources
        
        if ENABLE_WIKIPEDIA:
            tasks.append(search_wikipedia(claim, sources_per_type))
        if ENABLE_PUBMED:
            tasks.append(search_pubmed(claim, sources_per_type))
        if ENABLE_NEWSAPI:
            tasks.append(search_newsapi(claim, sources_per_type))
        if ENABLE_GOOGLE_SEARCH:
            tasks.append(search_google(claim, sources_per_type))
        
        # Social media sources
        if ENABLE_SOCIAL_MEDIA:
            await self.social_monitor.setup_apis()
            tasks.append(self.social_monitor.search_twitter(claim, sources_per_type))
            tasks.append(self.social_monitor.search_reddit(claim, sources_per_type))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_evidence = []
        for result in results:
            if isinstance(result, list):
                all_evidence.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Evidence gathering error: {result}")
        
        logger.info(f"Collected {len(all_evidence)} raw evidence pieces")
        
        # Add crisis context to evidence
        for evidence in all_evidence:
            evidence["crisis_context"] = relevant_crises
        
        unique_evidence = self.remove_duplicates(all_evidence)
        ranked_evidence = await rank_evidence_advanced(claim, unique_evidence)
        
        return ranked_evidence[:max_sources]

    def remove_duplicates(self, evidence_list: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for ev in evidence_list:
            title = ev.get("title", "").lower().strip()
            if title and title not in seen:
                seen.add(title)
                unique.append(ev)
        return unique

    def combine_verdicts_final(self, gemini_result: Dict, evidence_consensus: Dict) -> Tuple[str, float, str]:
        """Final verdict combination with evidence priority"""
        gemini_verdict = gemini_result.get("verdict", "INSUFFICIENT_INFO")
        gemini_confidence = gemini_result.get("confidence", 0.5)
        gemini_explanation = gemini_result.get("explanation", "")
        
        evidence_supports = evidence_consensus.get("supports", 0.0)
        evidence_refutes = evidence_consensus.get("refutes", 0.0)
        
        # Evidence-driven decision making with lower thresholds for better sensitivity
        if evidence_refutes > 0.5:  # Strong evidence refutation
            final_verdict = "FALSE"
            final_confidence = min(0.95, 0.75 + evidence_refutes * 0.2)
            explanation = f"Multiple sources contradict this claim. Evidence analysis shows {evidence_refutes:.1%} refutation vs {evidence_supports:.1%} support."
        elif evidence_supports > 0.5:  # Strong evidence support
            final_verdict = "TRUE"
            final_confidence = min(0.95, 0.75 + evidence_supports * 0.2)
            explanation = f"Multiple reliable sources confirm this claim. Evidence analysis shows {evidence_supports:.1%} support vs {evidence_refutes:.1%} refutation."
        elif evidence_refutes > evidence_supports and evidence_refutes > 0.25:
            final_verdict = "FALSE"
            final_confidence = min(0.85, 0.65 + evidence_refutes * 0.2)
            explanation = f"Evidence leans towards contradicting this claim. Analysis shows {evidence_refutes:.1%} refutation vs {evidence_supports:.1%} support."
        elif evidence_supports > evidence_refutes and evidence_supports > 0.25:
            final_verdict = "TRUE"
            final_confidence = min(0.85, 0.65 + evidence_supports * 0.2)
            explanation = f"Evidence leans towards supporting this claim. Analysis shows {evidence_supports:.1%} support vs {evidence_refutes:.1%} refutation."
        elif gemini_verdict == "FALSE":
            final_verdict = "FALSE"
            final_confidence = gemini_confidence
            explanation = gemini_explanation
        elif gemini_verdict == "TRUE":
            final_verdict = "TRUE"
            final_confidence = gemini_confidence
            explanation = gemini_explanation
        else:
            final_verdict = gemini_verdict if gemini_verdict != "INSUFFICIENT_INFO" else "INSUFFICIENT_INFO"
            final_confidence = gemini_confidence
            explanation = gemini_explanation
        
        return final_verdict, final_confidence, explanation

    async def fact_check_enhanced(self, claim: str, max_evidence: int = 12, monitor_social: bool = True, crisis_aware: bool = True) -> Dict:
        """Enhanced fact-checking with crisis awareness"""
        start_time = time.time()
        
        cache_key = hashlib.md5(f"{claim}_{max_evidence}_{monitor_social}_{crisis_aware}".encode()).hexdigest()
        if get_env_bool("ENABLE_CACHING", True) and cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < CACHE_TTL:
                logger.info("Using cached result")
                return cached["result"]
        
        # Step 1: Get crisis context
        crisis_context = []
        if crisis_aware and ENABLE_CRISIS_CONTEXT:
            crisis_context = self.crisis_db.get_relevant_crises(claim)
            logger.info(f"Found {len(crisis_context)} relevant crises")
        
        # Step 2: Enhanced evidence gathering
        logger.info("Gathering evidence from multiple sources...")
        evidence = await self.gather_evidence_enhanced(claim, max_evidence)
        
        # Step 3: Analyze evidence consensus
        logger.info("Analyzing evidence consensus...")
        evidence_consensus = analyze_evidence_consensus_enhanced(evidence, claim)
        
        # Step 4: Get Gemini analysis with crisis context
        logger.info("Getting crisis-aware Gemini analysis...")
        gemini_result = await gemini_fact_check_with_evidence_override(claim, evidence_consensus, crisis_context)
        
        # Step 5: Combine verdicts with evidence priority
        final_verdict, final_confidence, explanation = self.combine_verdicts_final(gemini_result, evidence_consensus)
        
        processing_time = time.time() - start_time
        
        result = {
            "claim": claim,
            "verdict": final_verdict,
            "confidence": round(final_confidence, 2),
            "explanation": explanation,
            "evidence": [EvidenceItem(**ev) for ev in evidence],
            "gemini_verdict": gemini_result.get("verdict", "INSUFFICIENT_INFO"),
            "evidence_consensus": evidence_consensus,
            "processing_time": round(processing_time, 2),
            "crisis_context": crisis_context,
        }
        
        if get_env_bool("ENABLE_CACHING", True):
            self.cache[cache_key] = {"result": result, "timestamp": time.time()}
        
        return result

# --- INITIALIZE GLOBAL COMPONENTS ---
fact_checker = EnhancedFactChecker()
stream_processor = MisinformationStreamProcessor()
trend_detector = TrendDetector()
demo_scenarios = DemoScenarios()

def verdict_display(verdict: str) -> str:
    """Return a user-friendly display for the verdict."""
    mapping = {
        "TRUE": "✅ TRUE",
        "FALSE": "❌ FALSE",
        "PARTIALLY_TRUE": "🟡 PARTIALLY TRUE",
        "INSUFFICIENT_INFO": "⚪ INSUFFICIENT INFO",
    }
    return mapping.get(verdict.upper(), verdict)

# --- API ENDPOINTS ---
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting Crisis Misinformation Detection System...")
    await stream_processor.setup_redis()
    await fact_checker.social_monitor.setup_apis()
    logger.info("System initialization complete")

@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check_endpoint(request: FactCheckRequest):
    """Legacy fact-check endpoint for backward compatibility"""
    analysis_id = hashlib.md5(request.claim.encode()).hexdigest()
    try:
        logger.info(f"Fact-checking request: {request.claim[:100]}...")
        result = await fact_checker.fact_check_enhanced(request.claim, request.max_evidence_sources, False, False)
        
        # Add the display field to the response
        result["final_verdict_display"] = verdict_display(result["verdict"])
        
        response = FactCheckResponse(analysis_id=analysis_id, **result)
        logger.info(f"Fact-check completed: {result['verdict']} ({result['confidence']})")
        return response
    except Exception as e:
        logger.exception(f"Fact-check error: {e}")
        raise HTTPException(500, f"Fact-check failed: {str(e)}")

@app.post("/crisis-fact-check", response_model=EnhancedFactCheckResponse)
async def crisis_fact_check_endpoint(request: CrisisFactCheckRequest):
    """Complete crisis-aware fact-checking with all features"""
    analysis_id = hashlib.md5(request.claim.encode()).hexdigest()
    try:
        logger.info(f"Crisis fact-checking request: {request.claim[:100]}...")
        start_time = time.time()
        
        # Enhanced fact-checking
        result = await fact_checker.fact_check_enhanced(
            request.claim,
            request.max_evidence_sources,
            request.monitor_social_media,
            request.crisis_context
        )
        
        # Add to trend detector for pattern analysis
        if ENABLE_TREND_DETECTION:
            trend_detector.add_content(request.claim)
        
        # Detect emerging patterns
        emerging_clusters = trend_detector.detect_emerging_clusters() if ENABLE_TREND_DETECTION else []
        
        # Calculate comprehensive risk assessment
        risk_indicators = {
            "viral_potential": calculate_viral_potential(result["evidence"]),
            "crisis_severity": max([c.get("severity_score", 0) for c in result.get("crisis_context", [])] + [0]),
            "misinformation_risk": calculate_misinformation_risk(result["evidence"], emerging_clusters),
            "coordination_risk": 0.3,  # Would analyze user patterns in full implementation
            "trend_velocity": len([c for c in emerging_clusters if c["risk_score"] > 0.7])
        }
        
        # Generate intervention recommendations
        interventions = generate_intervention_recommendations(
            result["verdict"],
            risk_indicators,
            result.get("crisis_context", [])
        )
        
        # Compile complete response
        enhanced_result = {
            "analysis_id": analysis_id,
            "final_verdict_display": verdict_display(result["verdict"]),
            "social_media_mentions": len([e for e in result["evidence"] if hasattr(e, 'source') and e.source in ["twitter", "reddit"]]),
            "trend_indicators": {
                "emerging_patterns": len(emerging_clusters),
                "high_risk_patterns": len([c for c in emerging_clusters if c["risk_score"] > 0.7]),
                "content_velocity": sum(c["velocity"] for c in emerging_clusters) / max(len(emerging_clusters), 1)
            },
            "risk_indicators": risk_indicators,
            "intervention_recommendations": interventions,
            **result
        }
        
        response = EnhancedFactCheckResponse(**enhanced_result)
        logger.info(f"Crisis fact-check completed: {result['verdict']} ({result['confidence']}) in {time.time() - start_time:.2f}s")
        return response
    except Exception as e:
        logger.exception(f"Crisis fact-check error: {e}")
        raise HTTPException(500, f"Crisis fact-check failed: {str(e)}")

# --- REAL-TIME MONITORING ENDPOINTS ---
@app.post("/start-monitoring")
async def start_stream_monitoring(keywords: List[str], background_tasks: BackgroundTasks):
    """Start monitoring streams for specific keywords"""
    if not ENABLE_REAL_TIME_MONITORING:
        raise HTTPException(400, "Real-time monitoring is disabled")
    
    logger.info(f"Starting monitoring for keywords: {keywords}")
    
    # Start monitoring in background
    background_tasks.add_task(stream_processor.start_monitoring_streams, keywords)
    
    return {"status": "monitoring_started", "keywords": keywords, "timestamp": datetime.now().isoformat()}

@app.get("/trends")
async def get_current_trends(time_window: int = 60):
    """Get current misinformation trends"""
    trends = await stream_processor.get_trend_analysis(time_window)
    return trends

@app.get("/alerts")
async def get_active_alerts():
    """Get active misinformation alerts"""
    alerts = await stream_processor.get_active_alerts()
    return {"alerts": alerts, "count": len(alerts), "timestamp": datetime.now().isoformat()}

@app.post("/analyze-trends")
async def analyze_current_trends():
    """Analyze current content trends for misinformation patterns"""
    if not ENABLE_TREND_DETECTION:
        raise HTTPException(400, "Trend detection is disabled")
    
    clusters = trend_detector.detect_emerging_clusters()
    return {
        "emerging_clusters": clusters,
        "high_risk_clusters": [c for c in clusters if c["risk_score"] > 0.7],
        "coordination_analysis": {"message": "Requires user activity data for full analysis"},
        "total_content_analyzed": len(trend_detector.recent_content),
        "analysis_timestamp": datetime.now().isoformat()
    }

# --- DEMO ENDPOINTS ---
@app.get("/demo/scenarios")
async def get_demo_scenarios():
    """Get all available demo scenarios"""
    return demo_scenarios.get_all_scenarios()

@app.post("/demo/run/{scenario_name}")
async def run_demo_scenario(scenario_name: str):
    """Run a specific demo scenario"""
    result = demo_scenarios.run_demo_scenario(scenario_name)
    return result

@app.get("/demo/dashboard")
async def demo_dashboard():
    """Get demo dashboard data"""
    return {
        "active_alerts": [
            {
                "alert_id": "demo_001",
                "keyword": "vaccine microchip",
                "severity": "high",
                "mentions": 847,
                "growth_rate": "320% in 2 hours",
                "status": "investigating"
            },
            {
                "alert_id": "demo_002",
                "keyword": "deepfake zelensky",
                "severity": "critical",
                "mentions": 1205,
                "growth_rate": "890% in 1 hour",
                "status": "confirmed_misinformation"
            }
        ],
        "trending_topics": [
            ("vaccine side effects", 1250),
            ("election fraud", 987),
            ("climate hoax", 743),
            ("deepfake detection", 521),
            ("fact check", 398)
        ],
        "system_stats": {
            "total_claims_analyzed": 45782,
            "misinformation_detected": 3891,
            "accuracy_rate": "94.7%",
            "average_detection_time": "1.3 minutes",
            "active_crises": len(fact_checker.crisis_db.active_crises),
            "monitoring_keywords": 25,
            "real_time_sources": 6
        },
        "capabilities": {
            "real_time_monitoring": ENABLE_REAL_TIME_MONITORING,
            "crisis_awareness": ENABLE_CRISIS_CONTEXT,
            "social_media_integration": ENABLE_SOCIAL_MEDIA,
            "trend_detection": ENABLE_TREND_DETECTION
        }
    }

# --- SYSTEM STATUS ENDPOINTS ---
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "v7.0",
        "environment": ENVIRONMENT,
        "components": {
            "gemini": "active" if GEMINI_API_KEY else "disabled",
            "evidence_sources": {
                "wikipedia": ENABLE_WIKIPEDIA,
                "pubmed": ENABLE_PUBMED,
                "newsapi": ENABLE_NEWSAPI and bool(NEWSAPI_KEY),
                "google_search": ENABLE_GOOGLE_SEARCH and bool(GOOGLE_API_KEY) and bool(GOOGLE_CX),
                "twitter": ENABLE_SOCIAL_MEDIA and bool(TWITTER_BEARER_TOKEN),
                "reddit": ENABLE_SOCIAL_MEDIA and bool(REDDIT_CLIENT_ID),
            },
            "crisis_features": {
                "real_time_monitoring": ENABLE_REAL_TIME_MONITORING,
                "trend_detection": ENABLE_TREND_DETECTION,
                "crisis_context": ENABLE_CRISIS_CONTEXT,
                "social_media": ENABLE_SOCIAL_MEDIA,
            },
            "infrastructure": {
                "redis": stream_processor.redis_client is not None,
                "vector_search": "faiss",
                "keyword_search": "bm25",
                "caching": get_env_bool("ENABLE_CACHING", True),
            },
        },
        "capabilities": [
            "Multi-source evidence aggregation",
            "Crisis-aware analysis",
            "Real-time social media monitoring",
            "Misinformation trend detection",
            "Proactive alert system",
            "Intervention recommendations"
        ]
    }

@app.get("/stats")
async def get_stats():
    return {
        "cache_size": len(fact_checker.cache) if get_env_bool("ENABLE_CACHING", True) else 0,
        "source_credibility": BASE_SOURCE_CREDIBILITY,
        "model_info": {
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "approach": "crisis-aware evidence-first analysis",
        },
        "configuration": {
            "max_workers": MAX_WORKERS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "cache_ttl": CACHE_TTL,
            "retry_limit": RETRY_LIMIT,
        },
        "crisis_database": {
            "active_crises": len(fact_checker.crisis_db.active_crises),
            "crisis_types": list(fact_checker.crisis_db.active_crises.keys())
        },
        "monitoring": {
            "stream_buffer_size": len(stream_processor.trend_buffer),
            "monitoring_active": stream_processor.monitoring_active,
            "trend_content_count": len(trend_detector.recent_content)
        }
    }

@app.get("/config")
async def get_config():
    """Get current configuration (non-sensitive)"""
    return {
        "environment": ENVIRONMENT,
        "debug": DEBUG,
        "enabled_sources": {
            "wikipedia": ENABLE_WIKIPEDIA,
            "pubmed": ENABLE_PUBMED,
            "newsapi": ENABLE_NEWSAPI,
            "google_search": ENABLE_GOOGLE_SEARCH,
            "twitter": ENABLE_SOCIAL_MEDIA and bool(TWITTER_BEARER_TOKEN),
            "reddit": ENABLE_SOCIAL_MEDIA and bool(REDDIT_CLIENT_ID),
        },
        "enabled_features": {
            "real_time_monitoring": ENABLE_REAL_TIME_MONITORING,
            "trend_detection": ENABLE_TREND_DETECTION,
            "crisis_context": ENABLE_CRISIS_CONTEXT,
            "social_media": ENABLE_SOCIAL_MEDIA,
        },
        "source_weights": BASE_SOURCE_CREDIBILITY,
        "model_settings": {
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "performance": {
            "max_workers": MAX_WORKERS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "cache_ttl": CACHE_TTL,
            "retry_limit": RETRY_LIMIT,
        },
    }

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await stream_processor.stop_monitoring()
    await fact_checker.social_monitor.close()
    logger.info("System shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        workers=WORKERS,
        reload=RELOAD and ENVIRONMENT == "development"
    )