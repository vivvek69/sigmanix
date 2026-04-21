import sqlite3
from datetime import datetime
import os

DATABASE = 'chatbot_database.db'

def get_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize database with required tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Students table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        visitor_id TEXT UNIQUE NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        total_conversations INTEGER DEFAULT 0,
        average_rating REAL DEFAULT 0
    )
    ''')
    
    # Conversations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        visitor_id TEXT NOT NULL,
        query TEXT NOT NULL,
        response TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (visitor_id) REFERENCES students(visitor_id)
    )
    ''')
    
    # Feedback table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        visitor_id TEXT NOT NULL,
        rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
        comment TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (visitor_id) REFERENCES students(visitor_id)
    )
    ''')
    
    conn.commit()
    conn.close()

def get_or_create_student(visitor_id):
    """Get or create a student record"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM students WHERE visitor_id = ?', (visitor_id,))
    student = cursor.fetchone()
    
    if not student:
        cursor.execute(
            'INSERT INTO students (visitor_id) VALUES (?)',
            (visitor_id,)
        )
        conn.commit()
    
    conn.close()
    return student

def save_conversation(visitor_id, query, response):
    """Save a conversation"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        'INSERT INTO conversations (visitor_id, query, response) VALUES (?, ?, ?)',
        (visitor_id, query, response)
    )
    
    # Update conversation count
    cursor.execute(
        'UPDATE students SET total_conversations = total_conversations + 1 WHERE visitor_id = ?',
        (visitor_id,)
    )
    
    conn.commit()
    conn.close()

def save_feedback(visitor_id, rating, comment=''):
    """Save feedback from user"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        'INSERT INTO feedback (visitor_id, rating, comment) VALUES (?, ?, ?)',
        (visitor_id, rating, comment)
    )
    
    # Update average rating
    cursor.execute(
        'SELECT AVG(rating) FROM feedback WHERE visitor_id = ?',
        (visitor_id,)
    )
    avg_rating = cursor.fetchone()[0] or 0
    
    cursor.execute(
        'UPDATE students SET average_rating = ? WHERE visitor_id = ?',
        (avg_rating, visitor_id)
    )
    
    conn.commit()
    conn.close()

def get_student_analytics():
    """Get analytics for all students"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            visitor_id,
            created_at,
            total_conversations,
            average_rating
        FROM students
        ORDER BY created_at DESC
    ''')
    
    students = []
    for row in cursor.fetchall():
        students.append({
            'visitor_id': row[0],
            'created_at': row[1],
            'total_conversations': row[2],
            'average_rating': row[3]
        })
    
    conn.close()
    return students

def get_conversation_history(visitor_id, limit=50):
    """Get conversation history for a student"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT query, response, created_at FROM conversations WHERE visitor_id = ? ORDER BY created_at DESC LIMIT ?',
        (visitor_id, limit)
    )
    
    conversations = []
    for row in cursor.fetchall():
        conversations.append({
            'query': row[0],
            'response': row[1],
            'created_at': row[2]
        })
    
    conn.close()
    return conversations

def get_feedback_history(visitor_id):
    """Get feedback history for a student"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT rating, comment, created_at FROM feedback WHERE visitor_id = ? ORDER BY created_at DESC',
        (visitor_id,)
    )
    
    feedbacks = []
    for row in cursor.fetchall():
        feedbacks.append({
            'rating': row[0],
            'comment': row[1],
            'created_at': row[2]
        })
    
    conn.close()
    return feedbacks
