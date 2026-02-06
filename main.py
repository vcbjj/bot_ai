
# main_bot.py
import asyncio
from telegram import Update
from dialects_database import DialectDatabase
from arabic_model import ArabicChatModel
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import logging

class MultiDialectBot:
    def __init__(self):
        # إعداد التسجيل
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # تهيئة المكونات
        self.dialect_db = DialectDatabase()
        self.chat_model = ArabicChatModel()
        self.group_memories = {}  # ذاكرة لكل مجموعة
        
        # متعلمون تكيفيون لكل لهجة
        self.learners = {
            "iraqi": AdaptiveLearner("iraqi"),
            "khaleeji": AdaptiveLearner("khaleeji"),
            "egyptian": AdaptiveLearner("egyptian")
        }
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالجة الرسالة الواردة"""
        user_message = update.message.text
        chat_id = update.message.chat_id
        
        # كشف اللهجة
        dialect = self.dialect_db.detect_dialect(user_message)
        
        # استرجاع تاريخ المحادثة
        if chat_id not in self.group_memories:
            self.group_memories[chat_id] = {
                "dialect": dialect,
                "history": [],
                "context": []
            }
        
        memory = self.group_memories[chat_id]
        
        # توليد الرد
        response = self.chat_model.generate_response(
            text=user_message,
            dialect=dialect,
            history=memory["history"]
        )
        
        # تحسين الرد حسب اللهجة
        refined_response = self.refine_for_dialect(response, dialect)
        
        # إرسال الرد
        await update.message.reply_text(refined_response)
        
        # تحديث الذاكرة
        memory["history"].append(f"المستخدم: {user_message}")
        memory["history"].append(f"البوت: {refined_response}")
        
        # الاحتفاظ بآخر 10 رسائل فقط
        if len(memory["history"]) > 10:
            memory["history"] = memory["history"][-10:]
        
        # التعلم من التفاعل (يمكن قياس النجاح بعدد الردود أو التفاعلات)
        self.learners[dialect].learn_from_interaction(
            user_input=user_message,
            bot_response=refined_response,
            success_score=0.8  # يمكن حساب هذا بشكل ديناميكي
        )
    
    def refine_for_dialect(self, text: str, dialect: str) -> str:
        """تحسين النص ليناسب اللهجة المحددة"""
        dialect_data = self.dialect_db.dialects.get(dialect, {})
        
        # استبدال الكلمات العامة بكلمات اللهجة
        for local_word, standard_word in dialect_data.get("common_words", {}).items():
            if standard_word in text:
                # استبدال جزئي مع مراعاة السياق
                text = text.replace(f" {standard_word} ", f" {local_word} ")
        
        # إضافة تحيات اللهجة
        greetings = dialect_data.get("greetings", [])
        if greetings and not text.startswith(tuple(greetings)):
            import random
            if random.random() > 0.7:  # 70% فرصة لإضافة تحية
                text = f"{random.choice(greetings)}، {text}"
        
        return text

# تشغيل البوت
async def main():
    bot = MultiDialectBot()
    
    # إعداد بوت تليجرام
    application = Application.builder().token("7765636958:AAE42pTqH-QTNWY5lPDJMDzarVIxJ0Q6b7A").build()
    
    # إضافة معالج الرسائل
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    
    # بدء البوت
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    print("✅ البوت يعمل الآن!")
    
    # إبقاء البوت يعمل
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
