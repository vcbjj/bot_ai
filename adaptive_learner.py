# adaptive_learner.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

class AdaptiveLearner:
    def __init__(self, dialect: str):
        self.dialect = dialect
        self.learning_file = f"data/learned_{dialect}.pkl"
        self.learned_patterns = self.load_learned()
        
        # ناقل الكلمات للتعرف على الأنماط
        self.vectorizer = CountVectorizer(
            ngram_range=(1, 3),
            max_features=1000
        )
    
    def load_learned(self):
        """تحميل الأنماط المتعلمة"""
        if os.path.exists(self.learning_file):
            with open(self.learning_file, 'rb') as f:
                return pickle.load(f)
        return {"patterns": [], "responses": []}
    
    def save_learned(self):
        """حفظ الأنماط المتعلمة"""
        os.makedirs("data", exist_ok=True)
        with open(self.learning_file, 'wb') as f:
            pickle.dump(self.learned_patterns, f)
    
    def learn_from_interaction(self, user_input: str, bot_response: str, success_score: float):
        """التعلم من التفاعل الناجح"""
        if success_score > 0.7:  # إذا كان التفاعل ناجحًا
            # استخراج n-grams من المدخلات
            patterns = self.extract_patterns(user_input)
            
            # إضافة إلى الأنماط المتعلمة
            self.learned_patterns["patterns"].extend(patterns)
            self.learned_patterns["responses"].append({
                "pattern": patterns[0] if patterns else user_input[:50],
                "response": bot_response,
                "score": success_score
            })
            
            self.save_learned()
    
    def extract_patterns(self, text: str):
        """استخراج أنماط لغوية من النص"""
        # هنا يمكن إضافة خوارزميات أكثر تقدمًا
        words = text.split()
        patterns = []
        
        # إنشاء n-grams
        for n in range(1, min(4, len(words) + 1)):
            for i in range(len(words) - n + 1):
                patterns.append(" ".join(words[i:i+n]))
        
        return patterns