# main.py
import asyncio
import logging
import sys
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_log.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# استيراد المكونات
from arabic_model import ArabicChatModel
from dialects_database import DialectDatabase
from adaptive_learner import AdaptiveLearner

class MultiDialectBot:
    def __init__(self):
        logger.info("🚀 جارٍ تهيئة البوت المتعدد اللهجات...")
        
        # تهيئة المكونات
        self.dialect_db = DialectDatabase()
        logger.info("✅ تم تحميل قاعدة بيانات اللهجات")
        
        try:
            self.chat_model = ArabicChatModel()
            logger.info("✅ تم تحميل النموذج اللغوي")
        except Exception as e:
            logger.error(f"❌ فشل تحميل النموذج: {e}")
            raise
        
        # متعلمون تكيفيون لكل لهجة
        self.learners = {}
        for dialect in ["iraqi", "khaleeji", "egyptian", "standard_arabic"]:
            self.learners[dialect] = AdaptiveLearner(dialect)
        
        logger.info("✅ تم تهيئة نظام التعلم التكيفي")
        
        # ذاكرة المجموعات
        self.group_memories = {}
        self.max_history = 5  # عدد الرسائل المحفوظة في التاريخ
        
        logger.info("🎉 اكتمل تهيئة البوت!")
    
    async def process_message(self, message_text: str, group_id: str, user_id: str = None) -> str:
        """معالجة رسالة وإرجاع رد"""
        try:
            logger.info(f"📨 معالجة رسالة من المجموعة {group_id}: {message_text[:50]}...")
            
            # كشف اللهجة
            dialect = self.dialect_db.detect_dialect(message_text)
            logger.info(f"🌍 اللهجة المكتشفة: {dialect}")
            
            # استرجاع أو إنشاء ذاكرة المجموعة
            if group_id not in self.group_memories:
                self.group_memories[group_id] = {
                    "dialect": dialect,
                    "history": [],
                    "users": set(),
                    "last_active": datetime.now()
                }
            
            memory = self.group_memories[group_id]
            
            # إضافة المستخدم إذا كان موجودًا
            if user_id:
                memory["users"].add(user_id)
            
            # توليد الرد
            response = self.chat_model.generate_response(
                text=message_text,
                dialect=dialect,
                history=memory["history"]
            )
            
            # تحسين الرد حسب اللهجة
            refined_response = self.refine_for_dialect(response, dialect)
            logger.info(f"🤖 الرد المُولد: {refined_response[:50]}...")
            
            # تحديث الذاكرة
            memory["history"].append(f"المستخدم: {message_text}")
            memory["history"].append(f"البوت: {refined_response}")
            memory["last_active"] = datetime.now()
            
            # الاحتفاظ بآخر الرسائل فقط
            if len(memory["history"]) > self.max_history * 2:
                memory["history"] = memory["history"][-(self.max_history * 2):]
            
            # التعلم من التفاعل (محاكاة النجاح)
            self.learners[dialect].learn_from_interaction(
                user_input=message_text,
                bot_response=refined_response,
                success_score=0.8
            )
            
            return refined_response
            
        except Exception as e:
            logger.error(f"❌ خطأ في معالجة الرسالة: {e}")
            return "عفواً، حدث خطأ في معالجتي. الرجاء المحاولة مرة أخرى."
    
    def refine_for_dialect(self, text: str, dialect: str) -> str:
        """تحسين النص ليناسب اللهجة المحددة"""
        try:
            dialect_data = self.dialect_db.dialects.get(dialect, {})
            
            # إذا كانت اللهجة موجودة في قاعدة البيانات
            if dialect_data:
                # استبدال الكلمات العامة بكلمات اللهجة (محسّن)
                common_words = dialect_data.get("common_words", {})
                
                # تجزئة النص إلى كلمات مع الحفاظ على المسافات
                words = text.split()
                refined_words = []
                
                for word in words:
                    # تحويل الكلمة إلى صيغة يمكن مقارنتها
                    clean_word = word.strip('.,!?؛،')
                    
                    if clean_word in common_words.values():
                        # البحث عن الكلمة المحلية المقابلة
                        for local_word, std_word in common_words.items():
                            if std_word == clean_word:
                                refined_words.append(local_word)
                                break
                        else:
                            refined_words.append(word)
                    else:
                        refined_words.append(word)
                
                text = ' '.join(refined_words)
                
                # إضافة تحية عشوائية
                import random
                greetings = dialect_data.get("greetings", [])
                if greetings and random.random() > 0.5:
                    text = f"{random.choice(greetings)}، {text}"
            
            return text
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحسين اللهجة: {e}")
            return text
    
    def get_group_stats(self, group_id: str) -> dict:
        """الحصول على إحصائيات المجموعة"""
        if group_id in self.group_memories:
            memory = self.group_memories[group_id]
            return {
                "dialect": memory["dialect"],
                "message_count": len(memory["history"]) // 2,
                "user_count": len(memory["users"]),
                "last_active": memory["last_active"].strftime("%Y-%m-%d %H:%M:%S")
            }
        return {}
    
    def cleanup_inactive_groups(self, hours_inactive=24):
        """تنظيف المجموعات غير النشطة"""
        current_time = datetime.now()
        inactive_groups = []
        
        for group_id, memory in self.group_memories.items():
            time_diff = current_time - memory["last_active"]
            if time_diff.total_seconds() > hours_inactive * 3600:
                inactive_groups.append(group_id)
        
        for group_id in inactive_groups:
            del self.group_memories[group_id]
            logger.info(f"🧹 تم تنظيف المجموعة غير النشطة: {group_id}")
        
        return len(inactive_groups)


# نموذج استخدام البوت
async def demo_bot():
    """عرض توضيحي للبوت"""
    bot = MultiDialectBot()
    
    # رسائل تجريبية
    test_messages = [
        ("شلونك يخوان، شخباركم اليوم؟", "iraqi"),
        ("شحوالك يا الغالي، ايش اخبارك؟", "khaleeji"),
        ("ازيك عامل ايه النهارده؟", "egyptian"),
        ("كيف حالكم اليوم؟", "standard_arabic")
    ]
    
    for message, expected_dialect in test_messages:
        print(f"\n{'='*50}")
        print(f"📤 المستخدم: {message}")
        print(f"🌍 اللهجة المتوقعة: {expected_dialect}")
        
        response = await bot.process_message(
            message_text=message,
            group_id="test_group",
            user_id="test_user"
        )
        
        print(f"🤖 البوت: {response}")
        
        # انتظار قصير
        await asyncio.sleep(1)
    
    print(f"\n{'='*50}")
    print("📊 إحصائيات المجموعة التجريبية:")
    stats = bot.get_group_stats("test_group")
    for key, value in stats.items():
        print(f"  {key}: {value}")

# دالة التشغيل الرئيسية
async def main():
    """الدالة الرئيسية لتشغيل البوت"""
    print("🚀 بدء تشغيل البوت المتعدد اللهجات...")
    
    try:
        # إنشاء البوت
        bot = MultiDialectBot()
        
        # عرض توضيحي
        await demo_bot()
        
        # هنا يمكنك إضافة تكامل مع Telegram أو Discord
        # ... الكود الخاص بالمنصة
        
        print("\n✅ البوت جاهز للعمل!")
        
        # إبقاء البوت يعمل
        keep_alive = True
        while keep_alive:
            # هنا يمكنك إضافة منطق للاستماع للمدخلات
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 تم إيقاف البوت بواسطة المستخدم")
    except Exception as e:
        print(f"\n❌ خطأ غير متوقع: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
