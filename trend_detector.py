import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import networkx as nx

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
        if len(self.recent_content) < min_samples:
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
    
    def detect_coordinated_behavior(self, user_activity: List[Dict]) -> Dict:
        """Detect coordinated inauthentic behavior"""
        if len(user_activity) < 10:
            return {"coordinated": False, "confidence": 0.0}
        
        # Create user interaction network
        G = nx.Graph()
        
        for activity in user_activity:
            user_id = activity.get("user_id")
            if user_id:
                G.add_node(user_id, **activity)
        
        # Add edges for users posting similar content within short timeframes
        for i, act1 in enumerate(user_activity):
            for act2 in user_activity[i+1:]:
                similarity = self.content_similarity(act1.get("content", ""), act2.get("content", ""))
                time_diff = abs(act1.get("timestamp", 0) - act2.get("timestamp", 0))
                
                if similarity > 0.8 and time_diff < 3600:  # Similar content within 1 hour
                    G.add_edge(act1["user_id"], act2["user_id"], weight=similarity)
        
        # Analyze network properties
        if len(G.edges()) == 0:
            return {"coordinated": False, "confidence": 0.0}
        
        # Calculate coordination indicators
        avg_clustering = nx.average_clustering(G)
        connected_components = list(nx.connected_components(G))
        largest_component_size = max(len(comp) for comp in connected_components) if connected_components else 0
        
        coordination_score = (avg_clustering * 0.5) + (largest_component_size / len(G) * 0.5)
        
        return {
            "coordinated": coordination_score > 0.6,
            "confidence": coordination_score,
            "network_size": len(G),
            "largest_component": largest_component_size,
            "clustering_coefficient": avg_clustering
        }

# Initialize trend detector
trend_detector = TrendDetector()

# Add new endpoint
@app.post("/analyze-trends")
async def analyze_current_trends():
    """Analyze current content trends for misinformation patterns"""
    clusters = trend_detector.detect_emerging_clusters()
    
    # Get recent user activity for coordination analysis
    # This would come from your social media monitoring
    coordination_analysis = {"message": "Requires user activity data"}
    
    return {
        "emerging_clusters": clusters,
        "high_risk_clusters": [c for c in clusters if c["risk_score"] > 0.7],
        "coordination_analysis": coordination_analysis,
        "total_content_analyzed": len(trend_detector.recent_content),
        "analysis_timestamp": datetime.now().isoformat()
    }
