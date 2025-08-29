import asyncio
import aiohttp
from typing import List, Dict
import tweepy
import praw
from datetime import datetime

class SocialMediaMonitor:
    def __init__(self):
        # Initialize API clients
        self.twitter_api = None
        self.reddit_api = None
        
    async def setup_apis(self):
        """Setup social media API clients"""
        # Twitter API v2 setup
        if os.getenv("TWITTER_BEARER_TOKEN"):
            self.twitter_client = tweepy.Client(
                bearer_token=os.getenv("TWITTER_BEARER_TOKEN")
            )
        
        # Reddit API setup
        if os.getenv("REDDIT_CLIENT_ID"):
            self.reddit_api = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent="CrisisMisinfoDetector/1.0"
            )
    
    async def search_twitter(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search recent tweets"""
        if not self.twitter_client:
            return []
        
        try:
            tweets = tweepy.Paginator(
                self.twitter_client.search_recent_tweets,
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics', 'author_id']
            ).flatten(limit=max_results)
            
            results = []
            for tweet in tweets:
                results.append({
                    "source": "twitter",
                    "title": f"Tweet by {tweet.author_id}",
                    "content": tweet.text,
                    "url": f"https://twitter.com/user/status/{tweet.id}",
                    "timestamp": tweet.created_at.isoformat(),
                    "engagement": tweet.public_metrics,
                    "platform_specific": {
                        "retweets": tweet.public_metrics['retweet_count'],
                        "likes": tweet.public_metrics['like_count']
                    }
                })
            return results
        except Exception as e:
            logger.error(f"Twitter search error: {e}")
            return []
    
    async def search_reddit(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Reddit posts"""
        if not self.reddit_api:
            return []
        
        try:
            results = []
            for submission in self.reddit_api.subreddit("all").search(query, limit=max_results):
                results.append({
                    "source": "reddit",
                    "title": submission.title,
                    "content": submission.selftext[:500] if submission.selftext else submission.title,
                    "url": f"https://reddit.com{submission.permalink}",
                    "timestamp": datetime.fromtimestamp(submission.created_utc).isoformat(),
                    "engagement": {
                        "upvotes": submission.score,
                        "comments": submission.num_comments
                    },
                    "platform_specific": {
                        "subreddit": submission.subreddit.display_name,
                        "upvote_ratio": submission.upvote_ratio
                    }
                })
            return results
        except Exception as e:
            logger.error(f"Reddit search error: {e}")
            return []
