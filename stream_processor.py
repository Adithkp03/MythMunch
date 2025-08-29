import asyncio
import json
from typing import Dict, List
from datetime import datetime
import aioredis
from collections import deque

class MisinformationStreamProcessor:
    def __init__(self):
        self.redis_client = None
        self.active_streams = {}
        self.trend_buffer = deque(maxlen=1000)  # Keep last 1000 items for trend analysis
        
    async def setup_redis(self):
        """Setup Redis connection for stream processing"""
        try:
            self.redis_client = await aioredis.from_url(
                "redis://localhost:6379", 
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def start_monitoring_streams(self, keywords: List[str]):
        """Start monitoring social media streams for keywords"""
        if not keywords:
            return
        
        # Create monitoring tasks for each keyword
        tasks = []
        for keyword in keywords:
            task = asyncio.create_task(self.monitor_keyword_stream(keyword))
            tasks.append(task)
            
        # Run all monitoring tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def monitor_keyword_stream(self, keyword: str):
        """Monitor a specific keyword stream"""
        logger.info(f"Starting stream monitoring for: {keyword}")
        
        while True:
            try:
                # Get recent mentions from social media
                recent_mentions = await self.get_recent_mentions(keyword)
                
                for mention in recent_mentions:
                    # Add to trend buffer
                    self.trend_buffer.append({
                        "keyword": keyword,
                        "mention": mention,
                        "timestamp": datetime.now().isoformat(),
                        "sentiment": await self.analyze_sentiment(mention["content"])
                    })
                    
                    # Check for suspicious patterns
                    if await self.detect_suspicious_pattern(mention):
                        await self.trigger_misinformation_alert(keyword, mention)
                
                # Wait before next check (adjust based on API limits)
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Stream monitoring error for {keyword}: {e}")
                await asyncio.sleep(60)  # Longer wait on error
    
    async def detect_suspicious_pattern(self, mention: Dict) -> bool:
        """Detect if a mention shows suspicious misinformation patterns"""
        suspicious_indicators = 0
        
        content = mention.get("content", "").lower()
        
        # Pattern 1: Urgent language
        urgent_words = ["breaking", "urgent", "immediately", "share now", "before they delete"]
        if any(word in content for word in urgent_words):
            suspicious_indicators += 1
        
        # Pattern 2: Conspiracy language
        conspiracy_words = ["they don't want you to know", "mainstream media hiding", "cover up"]
        if any(phrase in content for phrase in conspiracy_words):
            suspicious_indicators += 2
        
        # Pattern 3: Unverified claims
        unverified_words = ["sources say", "reportedly", "allegedly", "rumor has it"]
        if any(word in content for word in unverified_words):
            suspicious_indicators += 1
        
        # Pattern 4: High engagement on suspicious content
        engagement = mention.get("engagement", {})
        if engagement.get("shares", 0) > 1000 and suspicious_indicators > 0:
            suspicious_indicators += 2
        
        return suspicious_indicators >= 3
    
    async def trigger_misinformation_alert(self, keyword: str, mention: Dict):
        """Trigger alert for potential misinformation"""
        alert = {
            "alert_id": f"alert_{datetime.now().timestamp()}",
            "keyword": keyword,
            "mention": mention,
            "timestamp": datetime.now().isoformat(),
            "severity": "high",
            "status": "active"
        }
        
        # Store alert in Redis
        if self.redis_client:
            await self.redis_client.setex(
                f"alert:{alert['alert_id']}", 
                3600,  # 1 hour expiry
                json.dumps(alert)
            )
        
        # Log alert
        logger.warning(f"MISINFORMATION ALERT: {keyword} - {mention.get('content', '')[:100]}...")
        
        # You can add webhook notifications here
        await self.send_alert_notification(alert)
    
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
            }
        }

# Initialize stream processor
stream_processor = MisinformationStreamProcessor()

# Add new endpoints
@app.post("/start-monitoring")
async def start_stream_monitoring(keywords: List[str]):
    """Start monitoring streams for specific keywords"""
    await stream_processor.setup_redis()
    
    # Start monitoring in background
    asyncio.create_task(stream_processor.start_monitoring_streams(keywords))
    
    return {"status": "monitoring_started", "keywords": keywords}

@app.get("/trends")
async def get_current_trends():
    """Get current misinformation trends"""
    trends = await stream_processor.get_trend_analysis()
    return trends

@app.get("/alerts")
async def get_active_alerts():
    """Get active misinformation alerts"""
    if not stream_processor.redis_client:
        return {"alerts": []}
    
    # Get all alert keys
    alert_keys = await stream_processor.redis_client.keys("alert:*")
    alerts = []
    
    for key in alert_keys:
        alert_data = await stream_processor.redis_client.get(key)
        if alert_data:
            alerts.append(json.loads(alert_data))
    
    return {"alerts": alerts}
