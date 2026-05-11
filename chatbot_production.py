from flask import Flask, render_template_string, request, session, jsonify
from flask_cors import CORS
import os
import logging
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from database import (
    init_database,
    get_or_create_student,
    save_conversation,
    save_feedback,
    get_student_analytics,
    clear_user_data,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import re
from functools import wraps
from collections import defaultdict
import time

# ============ SETUP ============
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "sigmanix-secret-dev")

# Production-ready CORS configuration
# Default includes common React dev origin (Vite) and older CRA default
# Override by setting CORS_ORIGINS environment variable (comma-separated)
allowed_origins = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:8000,https:https://sigmanixtech.com"
).split(",") if o.strip()]

CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
    supports_credentials=True,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return '', 200


@app.after_request
def apply_dynamic_cors(response):
    """Ensure CORS headers are echoed for allowed origins (supports credentials)."""
    origin = request.headers.get("Origin")
    if origin and origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response
# Logging setup with UTF-8
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Database initialization
init_database()
logger.info("✅ Database initialized successfully")

# Groq API setup
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("❌ GROQ_API_KEY not found in environment variables")
else:
    groq_llm = ChatGroq(
        temperature=0.2,
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
    )
    logger.info("✅ Groq client initialized")

# Knowledge base setup
logger.info("📚 Loading knowledge base...")
try:
    with open("data.txt", "r", encoding="utf-8") as file:
        raw_text = file.read()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(raw_text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    logger.info(f"✅ Created {len(text_chunks)} chunks from knowledge base")
except Exception as e:
    logger.error(f"❌ Error loading knowledge base: {e}")
    knowledge_base = None
    text_chunks = []

# System prompt
SYSTEM_PROMPT = """You are a professional, concise chatbot for Sigmanix Tech.

Tone and style:
- Be polite, clear, and helpful.
- Keep a friendly but professional tone.
- Avoid slang, excessive emojis, hype, or overly casual wording.
- Use 2 to 3 short sentences.
- Do not write long paragraphs or lists unless the user asks for them.

Accuracy rules:
- Use only the facts provided in the retrieved context.
- Do not invent fees, discounts, placement numbers, salaries, or timings.
- If the answer is not in the context, say you are not sure and direct the user to the contact details.

Contact details:
- Phone: +91 7702476969
- Email: hr@sigmanixtech.com
- Location: Kondapur, Serilingampally, Hyderabad, Telangana - 500084
- Google Maps: https://maps.app.goo.gl/zCuvKLAZk7f4kprJ9?g_st=aw

Response goal:
- Answer the user's question directly.
- If helpful, mention the most relevant course, format, or next step.
- End with a simple question only when it fits naturally."""

# Menu responses
MENU_RESPONSES = {
    "courses": {
        "reply": "We offer AI-focused courses: Python with AI, Gen AI & Agentic AI, Data Analytics with AI, Prompt Engineering, Cybersecurity & Ethical Hacking, Agentic AI (Advanced), and Data Science with AI. Which course interests you?",
        "options": [
            {"label": "💻 Python & AI", "value": "Tell me about Python with AI course"},
            {"label": "🤖 Gen AI & Agents", "value": "What's in the Gen AI course?"},
            {"label": "📊 Data Analytics", "value": "Tell me about Data Analytics course"},
            {"label": "🔐 Cybersecurity", "value": "Tell me about Cybersecurity course"},
        ],
    },
    "duration": {
        "reply": "Course durations: Python with AI - 2 months, Gen AI & Agentic AI - 3 months, Data Analytics with AI - 2.5 months, Prompt Engineering - 6 weeks, Cybersecurity - 12 weeks, Agentic AI (Advanced) - flexible, Data Science with AI - 8 weeks. All courses are available in weekend, hybrid, fully online, and classroom formats. For exact batch timing, please contact +91 7702476969.",
        "options": [
            {"label": "🌙 Weekend Classes (Online)", "value": "Tell me more about weekend online classes"},
            {"label": "💻 Hybrid Classes", "value": "How do hybrid classes work?"},
            {"label": "📱 Fully Online", "value": "Can I study completely online anytime?"},
            {"label": "🏢 Classroom", "value": "Do you have classroom training at Hyderabad?"},
        ],
    },
    "placements": {
        "reply": "We provide job-ready training, real project experience, company referrals, interview preparation, resume support, and 1:1 mentorship. If you want, I can share the support offered for a specific course.",
        "options": [
            {"label": "📈 Success Stories", "value": "What are your placement rates?"},
            {"label": "🏢 Partner Companies", "value": "Which companies hire from you?"},
            {"label": "💪 Interview Prep", "value": "How do you prepare for interviews?"},
            {"label": "🎯 Job Roles", "value": "What jobs can I get after?"},
        ],
    },
    "registration": {
        "reply": "To register, submit the application form, speak with the admissions team, choose your course and batch, and then receive access after enrollment is confirmed. If you need help, contact +91 7702476969.",
        "options": [
            {"label": "📞 Contact Us", "value": "How do I contact your team?"},
            {"label": "❓ Requirements", "value": "What do I need to apply?"},
            {"label": "🎓 Prerequisites", "value": "Do I need prior experience?"},
            {"label": "🚀 Start Now", "value": "I want to enroll!"},
        ],
    },
    "menu": {
        "reply": "Welcome to Sigmanix Tech. How can I help you today? Choose an option below:",
        "options": [
            {"label": "📚 Courses", "value": "courses"},
            {"label": "⏱️ Duration & Timeline", "value": "duration"},
            {"label": "💼 Placements & Jobs", "value": "placements"},
            {"label": "📝 Registration & Admission", "value": "registration"},
            {"label": "❓ Other Questions", "value": "other"},
        ],
    },
}

# MOST ASKED QUESTIONS BY STUDENTS
MOST_ASKED_QUESTIONS = [
    "What will I learn in the Python with AI course?",
    "What's your placement success rate?",
    "How long does each course take?",
    "Can I attend classes if I'm working full-time?",
    "Do you offer weekend batches?",
    "What companies hire from Sigmanix Tech?",
    "Do I need prior programming experience?",
    "How much will this course cost?",
    "Can I study online or offline?",
    "What's the difference between your courses and other institutes?",
    "Will I get a certificate after completing?",
    "How often are the batches?",
    "What's the class schedule?",
    "Do you provide internship opportunities?",
    "Can I switch courses after starting?",
    "How do you prepare for interviews?",
    "What's the job placement assistance like?",
    "Are there scholarships available?",
    "What's the student to teacher ratio?",
    "Can I get a refund if I'm not satisfied?",
]

# Rate limiting
request_log = defaultdict(list)

def rate_limit(max_requests=20, window=60):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_id = request.remote_addr
            now = time.time()
            request_log[client_id] = [t for t in request_log[client_id] if now - t < window]
            if len(request_log[client_id]) >= max_requests:
                return jsonify({"error": "Rate limit exceeded. Try again later."}), 429
            request_log[client_id].append(now)
            return f(*args, **kwargs)
        return wrapper
    return decorator

def sanitize_response(text):
    return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)

def get_menu_response(menu_selected):
    return MENU_RESPONSES.get(menu_selected, None)

def detect_query_intent(query):
    """Detect if user is asking about having a query and suggest categories"""
    query_lower = query.lower()
    
    # Keywords that indicate user has a query/problem/need
    query_keywords = [
        "i have a question", "i have a query", "i need help", "i'm confused",
        "tell me about", "how do i", "can you help", "what about", "explain",
        "i want to know", "i'm interested", "help me", "can you tell me",
        "information about", "details about", "interested in"
    ]
    
    # Check if user is asking a query
    has_query = any(keyword in query_lower for keyword in query_keywords)
    
    if has_query:
        return True
    return False

def get_suggested_questions(query, response=None):
    """Get suggested follow-up questions based on user's query (like Amazon/Airtel)"""
    try:
        if not response:
            response = ""
        
        # Build context for suggestion
        context = f"User's question: {query[:100]}\nBot response: {response[:150]}"
        
        # Prompt to generate relevant follow-up questions
        prompt = f"""Based on this student question about Sigmanix Tech, generate 3 smart follow-up questions they might want to ask next. Make them specific and helpful.

{context}

Generate 3 engaging follow-up questions (max 10 words each) that naturally continue the conversation:
1. 
2. 
3. 
   
Format as simple questions without numbering."""
        
        response_text = groq_llm.invoke(prompt).content.strip()
        questions = []
        
        # Parse the response into individual questions
        for line in response_text.split('\n'):
            line = line.strip()
            if line:
                # Remove numbering if present
                line = re.sub(r'^\d+\.\s*', '', line).strip()
                if len(line) > 5 and len(line) < 80:  # Valid question length
                    questions.append({
                        "label": line,
                        "value": line
                    })
        
        return questions[:3]
    except Exception as e:
        logger.warning(f"Error generating suggestions: {e}")
        return []

def get_quick_suggestions(query):
    """Get quick category suggestions when user mentions query (like Amazon/Airtel)"""
    query_lower = query.lower()
    
    suggestions = []
    
    # Detect topic and suggest categories
    if any(word in query_lower for word in ["python", "ai", "data", "devops", "course", "learn", "training"]):
        suggestions = [
            {"label": "📚 Course Details", "value": "courses"},
            {"label": "⏱️ Duration & Fee", "value": "duration"},
            {"label": "💼 Job Support", "value": "placements"},
        ]
    elif any(word in query_lower for word in ["placement", "job", "salary", "career", "opportunity"]):
        suggestions = [
            {"label": "💼 Placement Support", "value": "placements"},
            {"label": "📚 Relevant Courses", "value": "courses"},
            {"label": "📝 How to Register", "value": "registration"},
        ]
    elif any(word in query_lower for word in ["register", "join", "enroll", "admission", "fee", "cost", "price"]):
        suggestions = [
            {"label": "📝 Registration Process", "value": "registration"},
            {"label": "💰 Fee & Payment", "value": "duration"},
            {"label": "📚 Choose Course", "value": "courses"},
        ]
    else:
        # Default suggestions
        suggestions = [
            {"label": "📚 Explore Courses", "value": "courses"},
            {"label": "💼 Career Path", "value": "placements"},
            {"label": "📝 Start Learning", "value": "registration"},
        ]
    
    return suggestions

def quick_reply(query):
    """Handle sensitive or high-confidence intents before LLM to reduce wrong replies."""
    q = (query or "").strip().lower()
    if not q:
        return None

    contact_line = "Call +91 7702476969"
    maps_link = "https://maps.app.goo.gl/zCuvKLAZk7f4kprJ9?g_st=aw"

    fee_words = [
        "fee", "fees", "price", "cost", "payment", "pay", "discount", "offer",
        "scholarship", "upfront", "emi", "installment", "group registration"
    ]
    timing_words = ["timing", "time", "schedule", "batch", "weekend", "weekday", "slot"]
    location_words = ["location", "address", "where", "office", "map", "directions", "visit"]
    bot_words = ["chatbot", "bot", "ai", "your responsibility", "your rules", "rules for ai", "responsibility of chatbot", "what can you do"]

    if any(word in q for word in fee_words):
        return {
            "reply": (
                "Fees, discounts, and payment plans depend on the course and batch. "
                f"Please {contact_line} for the latest offer and exact payment options."
            ),
            "options": [
                {"label": "📞 Contact Team", "value": "How do I contact your team?"},
                {"label": "📝 Registration", "value": "How do I register for a course?"},
                {"label": "📚 Course Details", "value": "courses"},
            ],
        }

    if any(word in q for word in timing_words):
        return {
            "reply": (
                "We offer weekend, hybrid, and fully online batches. Exact timings depend on the course and batch you choose, "
                f"so please {contact_line} for the current schedule."
            ),
            "options": [
                {"label": "⏱️ Duration & Formats", "value": "duration"},
                {"label": "📝 Registration", "value": "registration"},
                {"label": "📞 Contact Team", "value": "How do I contact your team?"},
            ],
        }

    if any(word in q for word in location_words):
        return {
            "reply": (
                "Our location is Kondapur, Serilingampally, Hyderabad, Telangana - 500084. "
                f"Google Maps: {maps_link}. For help, call +91 7702476969."
            ),
            "options": [
                {"label": "📍 Open Map", "value": "location"},
                {"label": "📚 Courses", "value": "courses"},
                {"label": "📞 Contact Team", "value": "How do I contact your team?"},
            ],
        }

    if "refund" in q or "money back" in q:
        return {
            "reply": (
                "Refund requests are reviewed by the admissions team on a case-by-case basis. "
                f"Please {contact_line} for the exact policy related to your enrollment."
            ),
            "options": [
                {"label": "📞 Contact Team", "value": "How do I contact your team?"},
                {"label": "📝 Registration", "value": "registration"},
            ],
        }

    if "python" in q and "ai" in q:
        return {
            "reply": (
                "Python with AI is a 2-month course covering Python fundamentals, machine learning, computer vision, hands-on projects, and deployment support. "
                "Would you like the syllabus or placement details?"
            ),
            "options": [
                {"label": "📚 Full Course List", "value": "courses"},
                {"label": "💼 Placement Support", "value": "placements"},
                {"label": "📝 Start Registration", "value": "registration"},
            ],
        }

    if "agentic ai" in q or "gen ai" in q:
        return {
            "reply": (
                "Gen AI & Agentic AI is a 3-month advanced course focused on practical AI workflows, tools, and real-world project work. "
                "Would you like the full topics breakdown?"
            ),
            "options": [
                {"label": "📚 Course Details", "value": "courses"},
                {"label": "⏱️ Duration", "value": "duration"},
                {"label": "💼 Placements", "value": "placements"},
            ],
        }

    return None

def generate_followup_questions(original_query, ai_response):
    """Generate AI-powered follow-up questions based on conversation context."""
    try:
        prompt = f"""You are a helpful assistant creating 3 smart follow-up questions for a student learning about tech courses.

Student's Question: {original_query}
Your Response: {ai_response}

Generate 3 specific, helpful follow-up questions that naturally continue this conversation. These should be questions the student might want to ask next. Keep them short (max 10 words each) and engaging.

Return ONLY the 3 questions, one per line, without numbering or bullet points. Example:
What is the average salary after Python?
Can I learn part-time?
Are there job placements?"""
        
        response = groq_llm.invoke(prompt)
        suggestions_text = response.content.strip()
        questions = []
        
        for line in suggestions_text.split('\n'):
            line = line.strip()
            if line and len(line) > 3 and len(line) < 100:  # Validate line
                # Remove any common prefixes
                line = re.sub(r'^[\d+.\-*]\s*', '', line).strip()
                if line:
                    questions.append({
                        "label": line,
                        "value": line
                    })
        
        # Return only first 3 questions
        return questions[:3]
    except Exception as e:
        logger.warning(f"Error generating follow-up questions: {e}")
        return []  # Return empty if there's an error

# ============ WEB UI ENDPOINT ============

@app.get("/")
def index():
    """Serve the chatbot HTML interface."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Sigmanix Tech Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }
        .container { width: 100%; max-width: 450px; height: 680px; background: white; border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); display: flex; flex-direction: column; overflow: hidden; animation: slideIn 0.3s ease-out; }
        @keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 24px 20px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25); }
        .header h1 { font-size: 22px; font-weight: 700; letter-spacing: 0.5px; margin: 0 0 4px 0; line-height: 1.3; }
        .header p { font-size: 13px; opacity: 0.9; margin: 0; font-weight: 500; }
        .chat-area { flex: 1; overflow-y: auto; padding: 16px; background: #f8f9fc; display: flex; flex-direction: column; gap: 12px; }
        .message-wrapper { display: flex; gap: 8px; margin-bottom: 4px; animation: fadeIn 0.25s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .bot-wrapper { justify-content: flex-start; }
        .user-wrapper { justify-content: flex-end; }
        .message { padding: 12px 14px; border-radius: 12px; max-width: 85%; font-size: 13px; line-height: 1.5; word-wrap: break-word; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        .user-msg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 14px 14px 4px 14px; font-weight: 500; }
        .bot-msg { background: #ffffff; color: #2c3e50; border-radius: 14px 14px 14px 4px; white-space: pre-wrap; word-break: break-word; border: 1px solid #e0e7ff; }
        .options-container { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 4px 0; width: 100%; animation: fadeIn 0.25s ease-out; }
        .options-btn { padding: 10px 14px; background: white; color: #667eea; border: 2px solid #667eea; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); flex: 0 1 auto; white-space: nowrap; }
        .options-btn:hover { background: #667eea; color: white; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3); }
        .menu-buttons { padding: 12px; background: white; border-top: 1px solid #e0e7ff; display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; flex-shrink: 0; }
        .menu-btn { padding: 12px; border: 2px solid #e0e7ff; background: white; border-radius: 10px; cursor: pointer; font-size: 22px; transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1); display: flex; align-items: center; justify-content: center; }
        .menu-btn:hover { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-color: #667eea; transform: scale(1.15); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25); }
        .menu-btn:active { transform: scale(1.05); }
        .input-area { padding: 14px; background: white; border-top: 1px solid #e0e7ff; display: flex; gap: 10px; flex-shrink: 0; }
        .input-area input { flex: 1; padding: 11px 14px; border: 2px solid #e0e7ff; border-radius: 8px; font-size: 13px; font-family: inherit; transition: all 0.25s; }
        .input-area input:focus { outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); }
        .input-area input::placeholder { color: #999; }
        .input-area button { padding: 11px 18px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 700; font-size: 16px; transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); display: flex; align-items: center; justify-content: center; min-width: 44px; height: 44px; }
        .input-area button:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3); }
        .input-area button:active { transform: translateY(0); }
        .input-area button:disabled { opacity: 0.6; cursor: not-allowed; }
        .chat-area::-webkit-scrollbar { width: 6px; }
        .chat-area::-webkit-scrollbar-track { background: transparent; }
        .chat-area::-webkit-scrollbar-thumb { background: #ddd; border-radius: 3px; }
        .chat-area::-webkit-scrollbar-thumb:hover { background: #999; }

        /* Feedback Modal */
        .modal-backdrop {
            position: fixed;
            inset: 0;
            background: rgba(15, 23, 42, 0.65);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 2000;
            padding: 20px;
        }
        .modal-backdrop.open { display: flex; }
        .feedback-modal {
            width: 100%;
            max-width: 420px;
            background: white;
            border-radius: 18px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
            overflow: hidden;
            animation: modalPop 0.22s ease-out;
        }
        @keyframes modalPop {
            from { opacity: 0; transform: translateY(14px) scale(0.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }
        .feedback-modal-header {
            padding: 18px 20px 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .feedback-modal-header h2 {
            font-size: 18px;
            margin: 0 0 6px 0;
        }
        .feedback-modal-header p {
            font-size: 13px;
            margin: 0;
            opacity: 0.92;
            line-height: 1.5;
        }
        .feedback-modal-body { padding: 18px 20px 20px 20px; }
        .star-row {
            display: flex;
            gap: 8px;
            justify-content: center;
            margin-bottom: 14px;
        }
        .star-btn {
            width: 46px;
            height: 46px;
            border: 1px solid #d8e0ff;
            background: #f8faff;
            border-radius: 12px;
            font-size: 24px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .star-btn:hover,
        .star-btn.active {
            transform: translateY(-2px) scale(1.04);
            background: #fff7d6;
            border-color: #f5c542;
        }
        .feedback-rating-label {
            text-align: center;
            font-size: 13px;
            color: #4b5563;
            margin-bottom: 14px;
            min-height: 20px;
        }
        .feedback-modal textarea {
            width: 100%;
            min-height: 96px;
            resize: vertical;
            border: 2px solid #e0e7ff;
            border-radius: 12px;
            padding: 12px 14px;
            font: inherit;
            font-size: 13px;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .feedback-modal textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.12);
        }
        .feedback-modal-actions {
            display: flex;
            gap: 10px;
            margin-top: 14px;
        }
        .feedback-modal-actions button {
            flex: 1;
            height: 44px;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .feedback-cancel-btn {
            background: #edf2ff;
            color: #344054;
        }
        .feedback-submit-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .feedback-modal-actions button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(102, 126, 234, 0.18);
        }
        .feedback-link {
            color: #2563eb;
            text-decoration: underline;
            word-break: break-word;
        }
        
        /* Loading Indicator */
        .typing-indicator { display: flex; gap: 4px; padding: 12px 14px; }
        .typing-dot { width: 8px; height: 8px; background: #667eea; border-radius: 50%; animation: bounce 1.4s infinite; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%, 80%, 100% { opacity: 0.3; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-8px); } }
        
        /* Improved Message Animations */
        .message { animation: messageSlide 0.35s cubic-bezier(0.34, 1.56, 0.64, 1); }
        @keyframes messageSlide { from { opacity: 0; transform: scale(0.8); } to { opacity: 1; transform: scale(1); } }
        
        /* Enhanced Options */
        .options-btn:active { transform: scale(0.95); }
        .options-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        /* Mobile Optimizations */
        @media (max-width: 768px) {
            .container { max-width: 95vw; height: 85vh; border-radius: 12px; }
            .header { padding: 16px 14px; }
            .header h1 { font-size: 20px; }
            .header p { font-size: 12px; }
            .chat-area { padding: 12px; gap: 10px; }
            .message { font-size: 14px; max-width: 90%; }
            .menu-buttons { grid-template-columns: repeat(5, 1fr); gap: 6px; padding: 10px; }
            .menu-btn { padding: 10px; font-size: 20px; border-radius: 8px; }
            .input-area { padding: 12px; gap: 8px; }
            .input-area input { padding: 10px 12px; font-size: 14px; }
            .input-area button { padding: 10px 16px; font-size: 18px; min-width: 40px; height: 40px; }
            .options-btn { padding: 9px 12px; font-size: 11px; }
        }
        
        @media (max-width: 480px) {
            .container { 
                width: 100vw; 
                height: 100vh; 
                max-width: 100vw; 
                border-radius: 0;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
            }
            .header { 
                padding: 14px 12px;
                border-radius: 0;
            }
            .header h1 { 
                font-size: 18px;
                margin-bottom: 2px;
            }
            .header p { 
                font-size: 11px;
            }
            .chat-area { 
                padding: 10px;
                gap: 8px;
            }
            .message { 
                font-size: 13px;
                max-width: 88%;
                padding: 10px 12px;
            }
            .message-wrapper { 
                gap: 6px;
                margin-bottom: 2px;
            }
            .menu-buttons { 
                grid-template-columns: repeat(5, 1fr);
                gap: 5px;
                padding: 8px;
                background: #f8f9fc;
            }
            .menu-btn { 
                padding: 8px;
                font-size: 18px;
                border-radius: 6px;
                min-height: 40px;
            }
            .input-area { 
                padding: 10px;
                gap: 8px;
                background: #f8f9fc;
            }
            .input-area input { 
                padding: 10px 10px;
                font-size: 14px;
                border-radius: 6px;
            }
            .input-area button { 
                padding: 10px;
                font-size: 18px;
                min-width: 44px;
                height: 44px;
                border-radius: 6px;
            }
            .options-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 6px;
                margin: 8px 0;
            }
            .options-btn { 
                padding: 10px 12px;
                font-size: 12px;
                border-radius: 6px;
                white-space: normal;
            }
        }
        
        /* Touch-friendly improvements */
        @media (hover: none) and (pointer: coarse) {
            .menu-btn:active { transform: scale(0.95); background: #667eea; color: white; }
            .options-btn:active { background: #667eea; color: white; }
            button { -webkit-tap-highlight-color: transparent; }
        }
        
        /* Improved responsiveness for large screens */
        @media (min-width: 769px) and (max-width: 1024px) {
            .container { max-width: 480px; }
        }
        
        @media (min-width: 1025px) {
            .container { max-width: 500px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Sigmanix Tech Chatbot</h1>
            <p>AI-Powered Career Assistant</p>
        </div>
        <div class="chat-area" id="chatArea"></div>
        <div class="menu-buttons">
            <button class="menu-btn" onclick="selectMenu('courses')" title="Courses">📚</button>
            <button class="menu-btn" onclick="selectMenu('duration')" title="Duration">⏱️</button>
            <button class="menu-btn" onclick="selectMenu('placements')" title="Placements">💼</button>
            <button class="menu-btn" onclick="selectMenu('registration')" title="Registration">📝</button>
            <button class="menu-btn" onclick="selectMenu('feedback')" title="Feedback">⭐</button>
            <button class="menu-btn" onclick="goBack()" title="Go Back">⬅️</button>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Ask a question..." onkeypress="handleEnter(event)">
            <button onclick="sendMessage()" title="Send">↑</button>
        </div>
    </div>
    <div class="modal-backdrop" id="feedbackModal" aria-hidden="true">
        <div class="feedback-modal" role="dialog" aria-modal="true" aria-labelledby="feedbackTitle">
            <div class="feedback-modal-header">
                <h2 id="feedbackTitle">Rate your experience</h2>
                <p>Tap a star like a Play Store review, then add an optional comment.</p>
            </div>
            <div class="feedback-modal-body">
                <div class="star-row" id="starRow" aria-label="Star rating">
                    <button class="star-btn" type="button" data-rating="1" aria-label="1 star">☆</button>
                    <button class="star-btn" type="button" data-rating="2" aria-label="2 stars">☆</button>
                    <button class="star-btn" type="button" data-rating="3" aria-label="3 stars">☆</button>
                    <button class="star-btn" type="button" data-rating="4" aria-label="4 stars">☆</button>
                    <button class="star-btn" type="button" data-rating="5" aria-label="5 stars">☆</button>
                </div>
                <div class="feedback-rating-label" id="feedbackRatingLabel">Select a rating</div>
                <textarea id="feedbackComment" placeholder="Write a short comment (optional)"></textarea>
                <div class="feedback-modal-actions">
                    <button class="feedback-cancel-btn" type="button" onclick="closeFeedbackModal()">Cancel</button>
                    <button class="feedback-submit-btn" type="button" onclick="submitFeedbackRating()">Submit</button>
                </div>
            </div>
        </div>
    </div>
    <script>
        let isLoading = false;
        let selectedFeedbackRating = 0;

        const feedbackRatingCopy = {
            1: 'Very poor',
            2: 'Poor',
            3: 'Okay',
            4: 'Good',
            5: 'Excellent'
        };
        
        function setLoading(state) {
            isLoading = state;
            const btn = document.querySelector('.input-area button');
            const input = document.getElementById('userInput');
            btn.disabled = state;
            input.disabled = state;
            btn.style.opacity = state ? '0.6' : '1';
        }
        
        function showTypingIndicator() {
            const chatArea = document.getElementById('chatArea');
            const wrapper = document.createElement('div');
            wrapper.className = 'message-wrapper bot-wrapper';
            wrapper.id = 'typingIndicator';
            const typingDiv = document.createElement('div');
            typingDiv.className = 'typing-indicator';
            typingDiv.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
            wrapper.appendChild(typingDiv);
            chatArea.appendChild(wrapper);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        function removeTypingIndicator() {
            const indicator = document.getElementById('typingIndicator');
            if (indicator) indicator.remove();
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function linkifyText(text) {
            const escaped = escapeHtml(text);
            return escaped.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer" class="feedback-link">$1</a>');
        }

        function renderLinkifiedContent(element, message) {
            const html = linkifyText(message).replace(/\n/g, '<br>');
            element.innerHTML = html;
        }
        
        // Animated typing effect for bot messages
        async function typeMessage(message, element) {
            let index = 0;
            element.textContent = '';
            
            const typeSpeed = 15; // milliseconds per character
            
            return new Promise(resolve => {
                function type() {
                    if (index < message.length) {
                        element.textContent += message[index];
                        index++;
                        setTimeout(type, typeSpeed);
                    } else {
                        resolve();
                    }
                }
                type();
            });
        }

        function openFeedbackModal() {
            selectedFeedbackRating = 0;
            document.getElementById('feedbackComment').value = '';
            document.getElementById('feedbackRatingLabel').textContent = 'Select a rating';
            updateFeedbackStars(0);
            document.getElementById('feedbackModal').classList.add('open');
            document.getElementById('feedbackModal').setAttribute('aria-hidden', 'false');
        }

        function closeFeedbackModal() {
            document.getElementById('feedbackModal').classList.remove('open');
            document.getElementById('feedbackModal').setAttribute('aria-hidden', 'true');
        }

        function updateFeedbackStars(rating) {
            const stars = document.querySelectorAll('.star-btn');
            stars.forEach((btn) => {
                const value = parseInt(btn.dataset.rating, 10);
                const active = value <= rating;
                btn.classList.toggle('active', active);
                btn.textContent = active ? '★' : '☆';
            });
            const label = document.getElementById('feedbackRatingLabel');
            label.textContent = rating ? `${rating}/5 - ${feedbackRatingCopy[rating]}` : 'Select a rating';
        }

        async function submitFeedbackRating() {
            if (!selectedFeedbackRating) {
                displayMessage('Please choose a star rating before submitting.', 'bot');
                return;
            }

            const comment = document.getElementById('feedbackComment').value.trim();
            setLoading(true);
            try {
                await fetch('/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rating: selectedFeedbackRating, comment })
                });
                closeFeedbackModal();
                displayMessage(`⭐ Thanks for your ${selectedFeedbackRating}/5 review!`, 'bot');
            } catch (error) {
                displayMessage('❌ Error submitting feedback. Please try again.', 'bot');
            } finally {
                setLoading(false);
            }
        }

        document.addEventListener('click', (event) => {
            const starButton = event.target.closest('.star-btn');
            if (!starButton) return;
            selectedFeedbackRating = parseInt(starButton.dataset.rating, 10);
            updateFeedbackStars(selectedFeedbackRating);
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                closeFeedbackModal();
            }
        });
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message || isLoading) return;
            
            displayMessage(message, 'user');
            input.value = '';
            input.focus();
            
            setLoading(true);
            showTypingIndicator();
            
            try {
                const response = await fetch('/chat', { 
                    method: 'POST', 
                    headers: { 'Content-Type': 'application/json' }, 
                    body: JSON.stringify({ message: message }),
                    signal: AbortSignal.timeout(30000)
                });
                
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                
                const data = await response.json();
                removeTypingIndicator();
                await displayBotMessage(data.reply);
                if (data.options) displayOptions(data.options);
            } catch (error) {
                removeTypingIndicator();
                const errorMsg = error.name === 'AbortError' 
                    ? 'Request timed out. Please try again.' 
                    : 'Unable to connect. Please check your connection.';
                displayMessage('❌ ' + errorMsg, 'bot');
            } finally {
                setLoading(false);
            }
        }
        
        async function selectMenu(menu) {
            if (isLoading) return;
            
            if (menu === 'feedback') {
                openFeedbackModal();
                return;
            }
            
            setLoading(true);
            showTypingIndicator();
            
            try {
                const response = await fetch('/chat', { 
                    method: 'POST', 
                    headers: { 'Content-Type': 'application/json' }, 
                    body: JSON.stringify({ menu_selected: menu }) 
                });
                
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                
                const data = await response.json();
                removeTypingIndicator();
                await displayBotMessage(data.reply);
                if (data.options) displayOptions(data.options);
            } catch (error) {
                removeTypingIndicator();
                displayMessage('❌ Error loading menu. Please try again.', 'bot');
            } finally {
                setLoading(false);
            }
        }
        
        function displayMessage(message, type) {
            const chatArea = document.getElementById('chatArea');
            const wrapper = document.createElement('div');
            wrapper.className = `message-wrapper ${type}-wrapper`;
            wrapper.setAttribute('role', 'article');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${type}-msg`;
            msgDiv.textContent = message;
            wrapper.appendChild(msgDiv);
            chatArea.appendChild(wrapper);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        // Display bot message with animated typing effect
        async function displayBotMessage(message) {
            const chatArea = document.getElementById('chatArea');
            const wrapper = document.createElement('div');
            wrapper.className = 'message-wrapper bot-wrapper';
            wrapper.setAttribute('role', 'article');
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message bot-msg';
            msgDiv.textContent = '';
            wrapper.appendChild(msgDiv);
            chatArea.appendChild(wrapper);
            
            // Animate the typing
            await typeMessage(message, msgDiv);
            renderLinkifiedContent(msgDiv, message);
            
            // Scroll to bottom after typing is done
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        function displayOptions(options) {
            if (!options) return;
            const chatArea = document.getElementById('chatArea');
            const optionsDiv = document.createElement('div');
            optionsDiv.className = 'options-container';
            optionsDiv.setAttribute('role', 'group');
            options.forEach((opt, idx) => {
                const btn = document.createElement('button');
                btn.className = 'options-btn';
                btn.textContent = opt.label;
                btn.setAttribute('aria-label', opt.label);
                btn.onclick = () => selectMenuOption(opt.value);
                optionsDiv.appendChild(btn);
            });
            chatArea.appendChild(optionsDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        async function selectMenuOption(value) {
            if (isLoading) return;
            
            // Check if this is a follow-up question (contains spaces/punctuation) or a menu selection
            const isFollowupQuestion = value.length > 20 || value.includes(' ') && value.length > 10;
            
            if (isFollowupQuestion) {
                // Send as regular message (follow-up question)
                displayMessage(value, 'user');
                setLoading(true);
                showTypingIndicator();
                
                try {
                    const response = await fetch('/chat', { 
                        method: 'POST', 
                        headers: { 'Content-Type': 'application/json' }, 
                        body: JSON.stringify({ message: value }) 
                    });
                    
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    
                    const data = await response.json();
                    removeTypingIndicator();
                    await displayBotMessage(data.reply);
                    if (data.options && data.options.length > 0) displayOptions(data.options);
                } catch (error) {
                    removeTypingIndicator();
                    displayMessage('❌ Error loading response. Please try again.', 'bot');
                } finally {
                    setLoading(false);
                }
            } else {
                // Send as menu selection
                setLoading(true);
                showTypingIndicator();
                
                try {
                    const response = await fetch('/chat', { 
                        method: 'POST', 
                        headers: { 'Content-Type': 'application/json' }, 
                        body: JSON.stringify({ menu_selected: value }) 
                    });
                    
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    
                    const data = await response.json();
                    removeTypingIndicator();
                    await displayBotMessage(data.reply);
                    if (data.options && data.options.length > 0) displayOptions(data.options);
                } catch (error) {
                    removeTypingIndicator();
                    displayMessage('❌ Error loading response. Please try again.', 'bot');
                } finally {
                    setLoading(false);
                }
            }
        }
        
        function handleEnter(event) {
            if (event.key === 'Enter' && !isLoading) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        function goBack() {
            const chatArea = document.getElementById('chatArea');
            if (chatArea.children.length > 0) {
                // Remove last bot message and its options
                const messages = chatArea.querySelectorAll('.message-wrapper');
                if (messages.length > 0) {
                    messages[messages.length - 1].remove();
                }
                const options = chatArea.querySelectorAll('.options-container');
                if (options.length > 0) {
                    options[options.length - 1].remove();
                }
            }
            // Show main menu
            selectMenu('menu');
        }
        
        // Initialize
        window.addEventListener('load', () => {
            selectMenu('menu');
            document.getElementById('userInput').focus();
        });
        
        // Auto-scroll to bottom when new messages arrive
        const observer = new MutationObserver(() => {
            const chatArea = document.getElementById('chatArea');
            chatArea.scrollTop = chatArea.scrollHeight;
        });
        observer.observe(document.getElementById('chatArea'), { childList: true });
    </script>
</body>
</html>"""
    return render_template_string(html)

# ============ CHAT ENDPOINT (for UI & React) ============

@app.route("/chat", methods=["POST", "OPTIONS"])
@rate_limit(max_requests=20, window=60)
def chat():
    """Main chat endpoint - works with UI and React."""
    try:
        if request.method == "OPTIONS":
            return ("", 200)

        payload = request.get_json(silent=True) or {}
        query = (payload.get("message") or "").strip()
        selected_menu = payload.get("menu_selected")

        if not query and not selected_menu:
            return jsonify({"reply": "Please type a question or select an option."}), 400

        # Initialize session
        if "visitor_id" not in session:
            session["visitor_id"] = 'visitor_' + os.urandom(12).hex()
            get_or_create_student(session["visitor_id"])
            logger.info(f"New visitor: {session['visitor_id']}")

        # Handle menu selection
        if selected_menu:
            menu_response = get_menu_response(selected_menu)
            if menu_response:
                save_conversation(
                    session["visitor_id"],
                    f"Menu: {selected_menu}",
                    menu_response["reply"],
                )
                return jsonify({
                    "reply": menu_response["reply"],
                    "options": menu_response.get("options", []),
                })

        # Try quick reply first
        quick = quick_reply(query)
        if quick:
            save_conversation(session["visitor_id"], query, quick["reply"])
            return jsonify({
                "reply": quick["reply"],
                "options": quick.get("options", []),
            })

        # Check if user has a query and offer suggestions (like Amazon/Airtel)
        has_query = detect_query_intent(query)
        
        # Use LLM for knowledge base search
        if knowledge_base is None:
            reply = "Knowledge base not loaded. Please try again later."
            followup_options = []
        else:
            query_result = knowledge_base.similarity_search(query, k=3)
            if not query_result:
                # If no exact match, still try to help with suggestions
                reply = "🤔 That's a great question! I want to make sure I give you the best answer.\n\nHere are some popular topics students ask about:"
                # If user has a query, show category suggestions
                if has_query:
                    followup_options = get_quick_suggestions(query)
                else:
                    followup_options = [
                        {"label": "📚 Courses", "value": "courses"},
                        {"label": "💼 Placements", "value": "placements"},
                        {"label": "📝 Registration", "value": "registration"},
                    ]
            else:
                # Combine retrieved documents as context
                context = "\n".join([doc.page_content for doc in query_result])
                prompt = f"""{SYSTEM_PROMPT}

Retrieved context:
{context}

User question: {query}

Write the response in a professional, natural style. Be concise, accurate, and specific to the context."""
                
                # Call LLM directly
                response = groq_llm.invoke(prompt)
                reply = sanitize_response(response.content.strip())
                
                # ALWAYS generate follow-up questions (like Amazon/Airtel)
                followup_options = generate_followup_questions(query, reply)
                
                # If AI-generated suggestions failed, fall back to smart categories
                if not followup_options:
                    if has_query:
                        followup_options = get_quick_suggestions(query)
                    else:
                        # Default smart suggestions
                        followup_options = [
                            {"label": "Tell me more 📚", "value": "Tell me more about this course"},
                            {"label": "Placement info 💼", "value": "What is the placement success rate?"},
                            {"label": "How to register 📝", "value": "How do I register for this course?"},
                        ]

        save_conversation(session["visitor_id"], query, reply)
        return jsonify({"reply": reply, "options": followup_options})
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"reply": "Error processing your request. Please try again."}), 500

@app.post("/feedback")
def feedback_endpoint():
    """Save user feedback."""
    try:
        if "visitor_id" not in session:
            return jsonify({"error": "Session not found"}), 400

        data = request.get_json(silent=True) or {}
        rating = data.get("rating", 0)
        comment = data.get("comment", "")

        if not (1 <= rating <= 5):
            return jsonify({"error": "Rating must be 1-5"}), 400

        save_feedback(session["visitor_id"], rating, comment)
        return jsonify({"success": True, "message": "Feedback saved"})
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({"error": "Error saving feedback"}), 500


@app.post("/reset")
def reset_endpoint():   
    """Clear current visitor's conversations and feedback."""
    try:
        if "visitor_id" not in session:
            return jsonify({"error": "Session not found"}), 400

        clear_user_data(session["visitor_id"])
        logger.info(f"Cleared data for visitor: {session['visitor_id']}")
        return jsonify({"success": True, "message": "User data cleared"})
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        return jsonify({"error": "Error clearing user data"}), 500

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running",
    })

# ============ ADMIN ANALYTICS ============

@app.get("/admin/students")
def get_students():
    """Get all students data."""
    try:
        analytics = get_student_analytics()
        return jsonify({"students": analytics})
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.get("/admin/analytics")
def get_analytics():
    """Get system analytics."""
    try:
        analytics = get_student_analytics()
        return jsonify({
            "total_students": len(analytics),
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ============ ERROR HANDLERS ============

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ============ MAIN ============

if __name__ == "__main__":
    logger.info("✅ Starting Sigmanix Chatbot...")
    logger.info("🌐 Server running on http://localhost:5000")
    logger.info("📚 Knowledge base ready with %d chunks", len(text_chunks))
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)