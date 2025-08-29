from datetime import datetime
from typing import Dict, List
import json

class CrisisDatabase:
    def __init__(self):
        self.active_crises = {
            "ukraine_war": {
                "name": "Ukraine-Russia Conflict",
                "keywords": ["ukraine", "russia", "putin", "zelensky", "kyiv", "moscow", "war", "invasion"],
                "entities": ["vladimir putin", "volodymyr zelensky", "ukraine", "russia"],
                "start_date": "2022-02-24",
                "status": "ongoing",
                "severity": "high"
            },
            "covid_pandemic": {
                "name": "COVID-19 Pandemic",
                "keywords": ["covid", "coronavirus", "vaccine", "pandemic", "lockdown", "mask"],
                "entities": ["who", "cdc", "pfizer", "moderna"],
                "start_date": "2020-01-01",
                "status": "ongoing", 
                "severity": "high"
            },
            "climate_crisis": {
                "name": "Climate Change",
                "keywords": ["climate change", "global warming", "carbon emissions", "renewable energy"],
                "entities": ["ipcc", "paris agreement", "cop28"],
                "start_date": "2000-01-01",
                "status": "ongoing",
                "severity": "high"
            }
        }
    
    def get_relevant_crises(self, claim: str) -> List[Dict]:
        """Find crises relevant to a claim"""
        claim_lower = claim.lower()
        relevant = []
        
        for crisis_id, crisis_data in self.active_crises.items():
            score = 0
            for keyword in crisis_data["keywords"]:
                if keyword in claim_lower:
                    score += 1
            for entity in crisis_data["entities"]:
                if entity.lower() in claim_lower:
                    score += 2
            
            if score > 0:
                crisis_data["relevance_score"] = score
                crisis_data["crisis_id"] = crisis_id
                relevant.append(crisis_data)
        
        return sorted(relevant, key=lambda x: x["relevance_score"], reverse=True)
