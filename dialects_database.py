# dialects_database.py
import json
from typing import Dict, List

class DialectDatabase:
    def __init__(self):
        self.dialects = {
            "iraqi": {
                "greetings": ["هلا", "شلونك", "ياهلا"],
                "common_words": {
                    "اكو": "يوجد",
                    "شني": "ماذا",
                    "شلون": "كيف",
                    "خل": "دع",
                    "هسه": "الآن"
                },
                "sentence_patterns": [
                    {"pattern": "شلون {}؟", "meaning": "كيف {}؟"}
                ]
            },
            "khaleeji": {
                "greetings": ["هلا والله", "شحوالك"],
                "common_words": {
                    "اشوفك": "أراك",
                    "ماجر": "فقط",
                    "عسب": "لأن",
                    "شدخل": "ما علاقة"
                }
            },
            "egyptian": {
                "greetings": ["ازيك", "اهلا"],
                "common_words": {
                    "يعني ايه": "ماذا يعني",
                    "عايز": "أريد",
                    "تمام": "حسنا"
                }
            }
        }
    
    def detect_dialect(self, text: str) -> str:
        """كشف اللهجة من النص"""
        scores = {}
        for dialect, data in self.dialects.items():
            score = 0
            for word in data["common_words"]:
                if word in text:
                    score += 1
            scores[dialect] = score
        
        return max(scores, key=scores.get)