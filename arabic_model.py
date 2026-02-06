# arabic_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class ArabicChatModel:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        """
        استخدام نموذج أصغر (1.5B) لأداء أفضل وتوافق أوسع
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ جارٍ تحميل النموذج على جهاز: {self.device}")
        
        try:
            # تحميل التوكينيزر أولاً
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # إذا لم يكن هناك GPU أو الذاكرة محدودة، استخدم التحميل بالـ CPU
            if self.device == "cpu" or not torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
            else:
                # استخدم quantization لتقليل حجم الذاكرة
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            # إنشاء pipeline للتفاعل
            self.chat_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print("✅ تم تحميل النموذج بنجاح!")
            
        except Exception as e:
            print(f"❌ خطأ في تحميل النموذج: {e}")
            # استخدام نموذج بدائي كبديل
            self.load_fallback_model()
    
    def load_fallback_model(self):
        """تحميل نموذج بسيط كبديل في حالة الفشل"""
        print("⚠️  جارٍ تحميل نموذج بديل...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("aubmindlab/aragpt2-base")
        self.model = GPT2LMHeadModel.from_pretrained("aubmindlab/aragpt2-base")
        
        if self.device == "cuda":
            self.model = self.model.cuda()
        
        # إعداد padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(self, text: str, dialect: str, history: list = None) -> str:
        """إنشاء رد مع مراعاة اللهجة"""
        try:
            # تحضير السياق
            history_text = "\n".join(history[-3:]) if history else ""
            
            # إعداد prompt مع اللهجة
            prompt = self.create_prompt(text, dialect, history_text)
            
            # توليد الرد
            if hasattr(self, 'chat_pipeline'):
                # استخدام pipeline للنماذج الكبيرة
                response = self.chat_pipeline(
                    prompt,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )[0]['generated_text']
            else:
                # للنماذب البسيطة
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # استخراج الرد فقط
            response = self.extract_response(response, prompt)
            return response
            
        except Exception as e:
            print(f"❌ خطأ في توليد الرد: {e}")
            return self.get_fallback_response(dialect)
    
    def create_prompt(self, text: str, dialect: str, history: str) -> str:
        """إنشاء prompt مناسب للهجة"""
        dialect_instructions = {
            "iraqi": "تحدث باللهجة العراقية العامية. استخدم كلمات مثل: شلونك، اكو، شني، خل، هسه.",
            "khaleeji": "تحدث باللهجة الخليجية. استخدم كلمات مثل: شحوالك، اشوفك، ماجر، عسب.",
            "egyptian": "تحدث باللهجة المصرية. استخدم كلمات مثل: ازيك، عايز، يعني ايه، تمام.",
            "standard_arabic": "تحدث بالعربية الفصحى المعاصرة."
        }
        
        instruction = dialect_instructions.get(dialect, dialect_instructions["standard_arabic"])
        
        prompt = f"""أنت بوت دردشة عربي يتحدث باللهجة {dialect}.
{instruction}

المحادثة السابقة:
{history}

المستخدم: {text}

البوت:"""
        
        return prompt
    
    def extract_response(self, full_text: str, prompt: str) -> str:
        """استخراج الرد من النص الكامل"""
        if "البوت:" in full_text:
            parts = full_text.split("البوت:")
            return parts[-1].strip()
        
        # إذا لم يتم العثور على "البوت:"، احذف الـ prompt
        if prompt in full_text:
            return full_text.replace(prompt, "").strip()
        
        return full_text.strip()
    
    def get_fallback_response(self, dialect: str) -> str:
        """رد افتراضي في حالة الخطأ"""
        responses = {
            "iraqi": "هلا والله، شلونك؟ شلون اساعدك؟",
            "khaleeji": "هلا والله، شحوالك؟ شو تحتاج؟",
            "egyptian": "اهلا، ازيك؟ ممكن اساعدك في ايه؟",
            "standard_arabic": "مرحباً، كيف يمكنني مساعدتك؟"
        }
        return responses.get(dialect, "مرحباً، كيف يمكنني مساعدتك؟")
