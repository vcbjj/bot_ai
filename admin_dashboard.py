# admin_dashboard.py
from flask import Flask, render_template, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/dashboard')
def dashboard():
    """لوحة تحكم لمراقبة أداء البوت"""
    conn = sqlite3.connect('bot_data.db')
    cursor = conn.cursor()
    
    # إحصائيات البوت
    stats = cursor.execute('''
        SELECT 
            COUNT(DISTINCT chat_id) as active_groups,
            COUNT(*) as total_messages,
            AVG(response_time) as avg_response_time,
            dialect,
            COUNT(*) as dialect_count
        FROM interactions 
        GROUP BY dialect
    ''').fetchall()
    
    return render_template('dashboard.html', stats=stats)

@app.route('/api/dialect_progress/<dialect>')
def dialect_progress(dialect):
    """تتبع تقدم تعلم اللهجة"""
    # بيانات لرسوم بيانية
    return jsonify({
        "dialect": dialect,
        "learned_words": 150,
        "interaction_count": 1200,
        "success_rate": 0.85
    })