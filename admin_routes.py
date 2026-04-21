from flask import Blueprint, jsonify
from database import get_connection

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/stats')
def get_stats():
    """Get chatbot statistics"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as total_chats FROM chat_sessions')
    total_chats = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(rating) as avg_rating FROM feedback')
    avg_rating_row = cursor.fetchone()
    avg_rating = avg_rating_row[0] if avg_rating_row and avg_rating_row[0] else 0
    
    cursor.execute('SELECT COUNT(*) as total_ratings FROM feedback')
    total_ratings = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'total_chats': total_chats,
        'average_rating': round(avg_rating, 2),
        'total_ratings': total_ratings
    })

@admin_bp.route('/recent-chats')
def get_recent_chats():
    """Get recent chat sessions"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, user_id, query, answer, timestamp FROM chat_sessions 
        ORDER BY timestamp DESC LIMIT 20
    ''')
    
    chats = []
    for row in cursor.fetchall():
        chats.append({
            'id': row[0],
            'user_id': row[1],
            'query': row[2],
            'answer': row[3],
            'timestamp': row[4]
        })
    
    conn.close()
    return jsonify({'chats': chats})

@admin_bp.route('/feedback')
def get_feedback():
    """Get all feedback"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT f.id, f.session_id, f.rating, f.feedback_text, f.timestamp 
        FROM feedback f
        ORDER BY f.timestamp DESC LIMIT 50
    ''')
    
    feedbacks = []
    for row in cursor.fetchall():
        feedbacks.append({
            'id': row[0],
            'session_id': row[1],
            'rating': row[2],
            'feedback_text': row[3],
            'timestamp': row[4]
        })
    
    conn.close()
    return jsonify({'feedbacks': feedbacks})
