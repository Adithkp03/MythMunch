import os
import re
import asyncio
import logging
import hashlib
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import requests
import faiss
import numpy as np
import aiohttp
import urllib.parse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from rank_bm25 import BM25Okapi

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

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

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

# Source Weights
WIKIPEDIA_WEIGHT = get_env_float("WIKIPEDIA_WEIGHT", 0.85)
PUBMED_WEIGHT = get_env_float("PUBMED_WEIGHT", 0.95)
NEWSAPI_WEIGHT = get_env_float("NEWSAPI_WEIGHT", 0.7)
GOOGLE_SEARCH_WEIGHT = get_env_float("GOOGLE_SEARCH_WEIGHT", 0.7)
GEMINI_WEIGHT = get_env_float("GEMINI_WEIGHT", 1.0)

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
logger = logging.getLogger("fact_checker")

# FastAPI app configuration
app = FastAPI(
    title="Truth Verification API",
    description="AI-powered fact-checking system with multi-source evidence analysis",
    version="v6.1",
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
}

dynamic_source_credibility = BASE_SOURCE_CREDIBILITY.copy()

# --- Pydantic Models ---
class FactCheckRequest(BaseModel):
    claim: str
    include_evidence: bool = True
    max_evidence_sources: Optional[int] = 8
    detailed_analysis: bool = False

class EvidenceItem(BaseModel):
    source: str
    title: str
    url: str
    snippet: str
    credibility: float
    similarity: float
    timestamp: str

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

# --- Retry Decorator for Resilience ---
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

# --- Evidence-Informed Gemini Fact-Checking ---
@async_retry()
async def gemini_fact_check_with_evidence_override(claim: str, evidence_consensus: Dict[str, float]) -> Dict[str, str]:
    """Enhanced fact-checking that considers evidence when making decisions"""
    # Build evidence summary for the prompt
    evidence_summary = ""
    if evidence_consensus:
        supports = evidence_consensus.get("supports", 0)
        refutes = evidence_consensus.get("refutes", 0)
        neutral = evidence_consensus.get("neutral", 0)
        evidence_summary = f"\n\nEVIDENCE SUMMARY:\n- Supporting evidence: {supports:.1%}\n- Contradicting evidence: {refutes:.1%}\n- Neutral evidence: {neutral:.1%}"

    prompt = f"""You are an expert fact-checker. Analyze this claim carefully:

CLAIM: "{claim}"

{evidence_summary}

Consider the evidence summary above when making your determination. If evidence strongly contradicts the claim (>70% refutation), the claim is likely FALSE. If evidence strongly supports the claim (>70% support), it's likely TRUE.

Important guidelines:
- Trust the evidence consensus when it's overwhelming (>70% in either direction)
- Pay attention to entity relationships (people, positions, countries)
- Consider current factual accuracy as of August 2025
- Be especially careful with political positions and geographical claims

Provide your analysis in this exact format:

VERDICT: [TRUE/FALSE/PARTIALLY_TRUE/INSUFFICIENT_INFO]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [Brief explanation considering both your knowledge and the evidence]
KEY_POINTS: [Main points supporting your verdict]

Be thorough but concise. Consider:
1. Your factual knowledge about the claim
2. The evidence summary provided above
3. Logical consistency and relationships between entities
4. Current and historical accuracy

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

# --- Evidence Retrieval Functions ---
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
                        abstract_match = re.search(r"<AbstractText>(.*?)</AbstractText>", xml, re.DOTALL)

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

# --- Advanced Ranking ---
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

# --- Enhanced Evidence Analysis ---
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
        strong_support = ["is the", "serves as", "currently", "has been", "elected as", "appointed as", "holds the position", "who is the"]
        moderate_support = ["confirms", "proves", "demonstrates", "shows", "supports", "validates", "since", "as of"]

        # Strong refutation indicators
        strong_refute = ["is not", "was never", "has never been", "incorrect", "false", "myth", "debunked", "not the"]
        moderate_refute = ["former", "ended", "resigned", "stepped down", "no longer", "was the"]

        # Specific logic for political positions
        if "prime minister" in claim_lower or "president" in claim_lower:
            # Extract key entities from claim
            claim_person = None
            claim_country = None
            claim_position = None

            # Simple entity extraction
            if "prime minister" in claim_lower:
                claim_position = "prime minister"
            elif "president" in claim_lower:
                claim_position = "president"

            # Check for USA/America references
            if any(term in claim_lower for term in ["usa", "united states", "america"]):
                claim_country = "usa"
            elif any(term in claim_lower for term in ["india", "indian"]):
                claim_country = "india"

            # Check for person names
            if "trump" in claim_lower:
                claim_person = "trump"
            elif "modi" in claim_lower:
                claim_person = "modi"

            # Analyze evidence text
            evidence_has_president = "president" in text
            evidence_has_pm = "prime minister" in text
            evidence_has_person = claim_person and claim_person in text
            evidence_has_country = claim_country and (
                (claim_country == "usa" and any(term in text for term in ["usa", "united states", "america", "american"])) or
                (claim_country == "india" and any(term in text for term in ["india", "indian"]))
            )

            # Direct position contradiction detection
            if claim_position == "prime minister" and evidence_has_president and not evidence_has_pm:
                if evidence_has_person or evidence_has_country:
                    is_refuting = True
            elif claim_position == "president" and evidence_has_pm and not evidence_has_president:
                if evidence_has_person or evidence_has_country:
                    is_refuting = True

            # Country-position mismatch detection
            if claim_country == "usa" and claim_position == "prime minister":
                # USA doesn't have PM
                if evidence_has_president and (evidence_has_person or any(term in text for term in ["usa", "united states", "america"])):
                    is_refuting = True

            # Positive confirmation detection
            if not is_refuting:
                if claim_position == "prime minister" and evidence_has_pm:
                    if evidence_has_person and evidence_has_country:
                        is_supporting = True
                    elif evidence_has_person or evidence_has_country:
                        if any(indicator in text for indicator in strong_support):
                            is_supporting = True
                elif claim_position == "president" and evidence_has_president:
                    if evidence_has_person and evidence_has_country:
                        is_supporting = True
                    elif evidence_has_person or evidence_has_country:
                        if any(indicator in text for indicator in strong_support):
                            is_supporting = True

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

# --- Main Fact Checker Class ---
class EnhancedFactChecker:
    def __init__(self):
        self.cache = {}

    async def gather_evidence(self, claim: str, max_sources: int = 8) -> List[Dict]:
        logger.info(f"Gathering evidence for claim: {claim[:50]}...")
        
        # Only call enabled sources
        tasks = []
        sources_per_type = max_sources // 4
        
        if ENABLE_WIKIPEDIA:
            tasks.append(search_wikipedia(claim, sources_per_type))
        if ENABLE_PUBMED:
            tasks.append(search_pubmed(claim, sources_per_type))
        if ENABLE_NEWSAPI:
            tasks.append(search_newsapi(claim, sources_per_type))
        if ENABLE_GOOGLE_SEARCH:
            tasks.append(search_google(claim, sources_per_type))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_evidence = []
        for result in results:
            if isinstance(result, list):
                all_evidence.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Evidence gathering error: {result}")

        logger.info(f"Collected {len(all_evidence)} raw evidence pieces")
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

    async def fact_check(self, claim: str, max_evidence: int = 8) -> Dict:
        start_time = time.time()
        cache_key = hashlib.md5(claim.encode()).hexdigest()

        if get_env_bool("ENABLE_CACHING", True) and cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < CACHE_TTL:
                logger.info("Using cached result")
                return cached["result"]

        # STEP 1: Gather evidence first
        logger.info("Gathering evidence from multiple sources...")
        evidence = await self.gather_evidence(claim, max_evidence)

        # STEP 2: Analyze evidence consensus
        logger.info("Analyzing evidence consensus...")
        evidence_consensus = analyze_evidence_consensus_enhanced(evidence, claim)

        # STEP 3: Get Gemini analysis WITH evidence context
        logger.info("Getting evidence-informed Gemini analysis...")
        gemini_result = await gemini_fact_check_with_evidence_override(claim, evidence_consensus)

        # STEP 4: Combine verdicts with evidence priority
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
        }

        if get_env_bool("ENABLE_CACHING", True):
            self.cache[cache_key] = {"result": result, "timestamp": time.time()}

        return result

fact_checker = EnhancedFactChecker()

def verdict_display(verdict: str) -> str:
    """Return a user-friendly display for the verdict."""
    mapping = {
        "TRUE": "✅ TRUE",
        "FALSE": "❌ FALSE",
        "PARTIALLY_TRUE": "🟡 PARTIALLY TRUE",
        "INSUFFICIENT_INFO": "⚪ INSUFFICIENT INFO",
    }
    return mapping.get(verdict.upper(), verdict)

@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check_endpoint(request: FactCheckRequest):
    analysis_id = hashlib.md5(request.claim.encode()).hexdigest()
    try:
        logger.info(f"Fact-checking request: {request.claim[:100]}...")
        result = await fact_checker.fact_check(request.claim, request.max_evidence_sources)
        
        # Add the display field to the response
        result["final_verdict_display"] = verdict_display(result["verdict"])
        
        response = FactCheckResponse(analysis_id=analysis_id, **result)
        logger.info(f"Fact-check completed: {result['verdict']} ({result['confidence']})")
        return response
    except Exception as e:
        logger.exception(f"Fact-check error: {e}")
        raise HTTPException(500, f"Fact-check failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "v6.1",
        "environment": ENVIRONMENT,
        "components": {
            "gemini": "active" if GEMINI_API_KEY else "disabled",
            "evidence_sources": {
                "wikipedia": ENABLE_WIKIPEDIA,
                "pubmed": ENABLE_PUBMED,
                "newsapi": ENABLE_NEWSAPI and bool(NEWSAPI_KEY),
                "google_search": ENABLE_GOOGLE_SEARCH and bool(GOOGLE_API_KEY) and bool(GOOGLE_CX),
            },
            "enhanced_analysis": get_env_bool("ENABLE_ENHANCED_ANALYSIS", True),
            "evidence_first_approach": "active",
            "vector_search": "faiss",
            "keyword_search": "bm25",
            "caching": get_env_bool("ENABLE_CACHING", True),
        },
    }

@app.get("/stats")
async def get_stats():
    return {
        "cache_size": len(fact_checker.cache) if get_env_bool("ENABLE_CACHING", True) else 0,
        "source_credibility": BASE_SOURCE_CREDIBILITY,
        "model_info": {
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "approach": "evidence-first with llm override",
        },
        "configuration": {
            "max_workers": MAX_WORKERS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "cache_ttl": CACHE_TTL,
            "retry_limit": RETRY_LIMIT,
        },
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT, 
        workers=WORKERS,
        reload=RELOAD and ENVIRONMENT == "development"
    )