from datetime import datetime
import json

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
                        "content": "BREAKING: Zelensky surrender video is FAKE! Digital forensics show clear deepfake artifacts.",
                        "credibility": 0.9,
                        "verification_status": "verified_debunk"
                    },
                    {
                        "source": "reuters", 
                        "content": "Fact Check: Video of Zelensky calling for surrender is a deepfake",
                        "credibility": 0.95,
                        "verification_status": "fact_checked_false"
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
                        "content": "Comprehensive analysis of mRNA vaccine ingredients shows no metallic components",
                        "credibility": 0.98,
                        "verification_status": "peer_reviewed"
                    },
                    {
                        "source": "cdc",
                        "content": "COVID-19 vaccines do not contain microchips or tracking devices",
                        "credibility": 0.97,
                        "verification_status": "official_health_authority"
                    }
                ],
                "expected_verdict": "FALSE", 
                "misinformation_type": "conspiracy_theory",
                "potential_harm": "medium"
            },
            
            "climate_hoax": {
                "name": "Climate Change Denial",
                "claim": "Global warming is a hoax created by scientists for funding",
                "actual_status": "SCIENTIFIC CONSENSUS - Climate change is real",
                "crisis_context": "climate_crisis",
                "demo_evidence": [
                    {
                        "source": "ipcc",
                        "content": "97% of climate scientists agree human activities are primary cause of climate change",
                        "credibility": 0.99,
                        "verification_status": "scientific_consensus"
                    },
                    {
                        "source": "nature",
                        "content": "Multiple independent studies confirm accelerating global temperature rise",
                        "credibility": 0.96,
                        "verification_status": "peer_reviewed"
                    }
                ],
                "expected_verdict": "FALSE",
                "misinformation_type": "science_denial", 
                "potential_harm": "high"
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
            "crisis_context": scenario["crisis_context"],
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

# Add demo endpoints
demo_scenarios = DemoScenarios()

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
            "average_detection_time": "1.3 minutes"
        }
    }
