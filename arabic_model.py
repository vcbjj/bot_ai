# arabic_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ArabicChatModel:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # تحميل النموذج المخصص للعربية
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # إعداد prompt template للعربية
        self.chat_template = """أنت بوت عربي يتكلم باللهجة {dialect}. 
المحادثة السابقة:
{history}
المستخدم: {input}
البوت:"""
    
    def generate_response(self, text: str, dialect: str, history: list = None) -> str:
        """إنشاء رد مع مراعاة اللهجة"""
        # تحضير السياق
        history_text = "\n".join(history[-5:]) if history else ""
        
        # إعداد prompt مع اللهجة
        prompt = self.chat_template.format(
            dialect=dialect,
            history=history_text,
            input=text
        )
        
        # توليد الرد
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # استخراج الرد فقط
        response = response.split("البوت:")[-1].strip()
        
        return response