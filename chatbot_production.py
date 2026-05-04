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
    "http://localhost:5173,http://localhost:3000,https://localhost:8000"
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
        temperature=0.7,
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
SYSTEM_PROMPT = """You're a friendly friend helping with Sigmanix Tech. KEEP IT CASUAL & HELPFUL! 😊

⚡ COMMUNICATION STYLE - MOST IMPORTANT:
- Speak like a REAL FRIEND chatting, not a bot
- USE CASUAL LANGUAGE: "yep", "totally", "yeah", "cool", "awesome"
- RESPONSE LENGTH: 2-3 short sentences MAX (add explanation, not big paragraphs!)
- CONVERSATIONAL: Ask questions, sound natural, make them want the course
- No fancy bullet points or formatting - just chat
- Emojis are okay but not overused
- Be warm but quick - like texting a buddy

💬 RESPONSE STYLE (SHORT BUT WITH EXPLANATION):
Good: "Oh awesome! Python with AI is a 2-month course where you'll learn AI, machine learning, and computer vision. You'll work on real projects and get job-ready skills that companies actually want right now! Perfect for breaking into tech 🚀"
Better: "Cool! Python with AI is a 2-month program covering AI, ML, and computer vision. You'll build real projects, get mentored by experts, and have companies reaching out to you. Honestly, it's the fastest way to land a great job in AI! 🚀"

💼 PERSUASION TACTICS:
- Highlight REAL BENEFITS: job-ready, hands-on, expert mentors, company referrals
- Make them see VALUE: "You'll learn X, which gets you Y job role with Z salary potential"
- Create URGENCY: "People are getting hired super fast from this batch"
- Show PROOF: "Our students are getting placed in top companies"
- Make it RELATABLE: "Just like you, most students come with no AI experience and land great jobs"

🎯 YOUR RESPONSIBILITIES:
1. Answer ALL questions honestly
2. Help them pick the RIGHT course for THEIR goals
3. Share REAL info only, no making stuff up
4. For FEES, DISCOUNTS, TIMINGS → Say: "That's something our team can customize for you! Contact them at +91 7702476969 - they're super helpful 😊"
5. Be genuine, persuasive, and quick
6. Guide them as a friend, not a sales bot

📞 CONTACT INFO:
Phone: +91 7702476969 | Email: hr@sigmanixtech.com | Location: Bangalore, India

🎓 MOST POPULAR COURSES:
Python with AI • Gen AI & Agentic AI • Data Analytics with AI • DevOps Multi-Cloud • Prompt Engineering • Cybersecurity • RPA UiPath • Salesforce Developer • SAP ABAP • Agentic AI

🌟 CLASS FORMATS (ALL AVAILABLE):
• Weekend Classes - FULLY ONLINE (Saturday & Sunday live sessions)
• Hybrid Classes - Mix of online & offline at Bangalore
• Fully Online - 24/7 access to live & recorded classes
• Classroom Training - In-person at Bangalore location
Faculty will provide specific timings upon enrollment!

⚠️ CONFIDENTIAL INFO - NEVER SHARE DIRECTLY:
❌ Specific fees/pricing (REDIRECT: "Our team customizes packages - call +91 7702476969")
❌ Specific discounts (REDIRECT: "Ask our admissions team about current offers!")
❌ Exact timings (REDIRECT: "Faculty shares timings after enrollment - depends on your preference")
❌ Unconfirmed job guarantees
❌ Made-up student salaries  

✅ INSTEAD DO THIS:
- When asked about FEES: "That depends on your course and preference! Our team can make you an amazing offer at +91 7702476969 💰"
- When asked about DISCOUNTS: "We have different offers for early birds, referrals, and students - contact our team to see what you qualify for!"
- When asked about TIMINGS: "You get to choose your batch timing! Once you apply, our team works with you to pick the perfect schedule"

RESPONSE LENGTH RULES:
- NEVER write long paragraphs (max 3 short sentences)
- NEVER be formal or robotic
- DO make explanations helpful but short
- DO sound like you're texting a friend
- DO use contractions: "it's", "you're", "we've"
- DO add WHY they should care (value proposition)

WHEN COMPARING WITH OTHER INSTITUTES:
- Say: "It's smart that you're exploring! Here's what makes us different..."
- Highlight: job-ready focus, direct company referrals, hands-on projects, expert mentors, fast learning, career support
- Never bad-mouth others

💡 REMEMBER:
Make responses SHORT but PACKED with value. Sound like a friend who genuinely wants to help them succeed - not a bot. 
Be warm, persuasive, honest, and quick. That's it! 🎯"""

# Menu responses
MENU_RESPONSES = {
    "courses": {
        "reply": "📚 **Available Courses:**\n🎓 Python with AI (2 months) - Break into AI with hands-on projects\n🎓 Gen AI & Agentic AI (3 months) - Learn the future of automation\n🎓 Data Analytics with AI (2.5 months) - Turn data into career opportunities\n🎓 DevOps Multi-Cloud (3 months) - Get hired as a DevOps engineer\n🎓 Cybersecurity • Prompt Engineering • RPA • Salesforce & more!\n\nWhich one excites you? 🚀",
        "options": [
            {"label": "💻 Python & AI", "value": "Tell me about Python with AI course"},
            {"label": "🤖 Gen AI & Agents", "value": "What's in the Gen AI course?"},
            {"label": "📊 Data Analytics", "value": "Tell me about Data Analytics course"},
            {"label": "🌐 DevOps Multi-Cloud", "value": "What will I learn in DevOps?"},
        ],
    },
    "duration": {
        "reply": "⏱️ **Course Durations & Formats:**\n• Python with AI: 2 months\n• Gen AI & Agentic AI: 3 months\n• Data Analytics with AI: 2.5 months\n• DevOps Multi-Cloud: 3 months\n• Prompt Engineering: 6 weeks\n• Cybersecurity: 12 weeks\n\n✨ **Class Formats Available:**\n🌙 Weekend Classes (ONLINE) - Saturday & Sunday live sessions\n💻 Hybrid Classes - Online + In-person at Bangalore\n📱 Fully Online - 24/7 access to live & recorded classes\n🏢 Offline/Classroom - In-person at Bangalore location\n\n⏰ For specific timings & batch schedules, our team will customize based on YOUR preference! Call +91 7702476969 😊",
        "options": [
            {"label": "🌙 Weekend Classes (Online)", "value": "Tell me more about weekend online classes"},
            {"label": "💻 Hybrid Classes", "value": "How do hybrid classes work?"},
            {"label": "📱 Fully Online", "value": "Can I study completely online anytime?"},
            {"label": "🏢 Classroom", "value": "Do you have classroom training at Bangalore?"},
        ],
    },
    "placements": {
        "reply": "💼 **Here's What We Do For Your Career:**\n✓ Job-Ready Training (you'll learn what companies actually want)\n✓ Real Project Experience (not just theory)\n✓ Direct Company Referrals (we know top companies)\n✓ Interview Prep & Mock Interviews (practice with pros)\n✓ Resume Review & Career Guidance (get noticed by recruiters)\n✓ 1:1 Mentorship (guidance from industry experts)\n\nOur students are getting placed in amazing companies! Want success stories? 🎯",
        "options": [
            {"label": "📈 Success Stories", "value": "What are your placement rates?"},
            {"label": "🏢 Partner Companies", "value": "Which companies hire from you?"},
            {"label": "💪 Interview Prep", "value": "How do you prepare for interviews?"},
            {"label": "🎯 Job Roles", "value": "What jobs can I get after?"},
        ],
    },
    "registration": {
        "reply": "📝 **Getting Started is SUPER Easy:**\n1️⃣ Apply on our website or fill a quick form\n2️⃣ Chat with our admissions team (they're awesome!)\n3️⃣ Pick your course, batch & timing\n4️⃣ Get course access within 24 hours - start learning! 🚀\n\n💥 Don't wait - batches fill up fast & new ones start soon!",
        "options": [
            {"label": "📞 Contact Us", "value": "How do I contact your team?"},
            {"label": "❓ Requirements", "value": "What do I need to apply?"},
            {"label": "🎓 Prerequisites", "value": "Do I need prior experience?"},
            {"label": "🚀 Start Now", "value": "I want to enroll!"},
        ],
    },
    "menu": {
        "reply": "Welcome to Sigmanix Tech! 👋\n\nHow can I help you today? Choose from below:",
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

# ============ WEB UI ENDPOINT ============

@app.get("/")
def index():
    """Serve the chatbot HTML interface."""
    html = """<!DOCTYPE html>
... (file truncated for brevity) ...
"""
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
                prompt = f"""You're a friendly guide for Sigmanix Tech. Answer BRIEFLY like texting a friend - keep it to 1-2 sentences MAX!

FACTS YOU CAN USE:
{context}

STUDENT ASKS: {query}

REPLY LIKE YOU'RE TEXTING A BUDDY - casual, short, warm! Use "yeah", "cool", "awesome", emojis okay but not too many. Be honest if unsure."""
                
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
