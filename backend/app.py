from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle, json, logging, os, re
import google.generativeai as genai
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
import subprocess
import psutil

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "defaultsecretkey")

# SQLAlchemy config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Logger
logging.basicConfig(level=logging.DEBUG)

# Base path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ========== User Model ==========
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Create DB if not exists
with app.app_context():
    db.create_all()

# ========== Load ML Models ==========
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE_DIR, "model")

    predictor_path = os.path.join(model_dir, "disease_predictor.pkl")
    vectorizer_path = os.path.join(model_dir, "symptom_vectorizer.pkl")

    with open(predictor_path, "rb") as f:
        predictor_model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    all_symptoms = vectorizer.get_feature_names_out().tolist()
    logging.info("✅ Models loaded successfully.")
except Exception as e:
    logging.error(f"Model/vectorizer load error: {e}")
    predictor_model = None
    vectorizer = None
    all_symptoms = []


# ========== Load Disease Info ==========
try:
    # If script is inside backend/, go one level up to find backend/data/
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "data", "disease_info.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    with open(data_path, "r", encoding='utf-8') as f:
        disease_info_db = json.load(f)
    logging.info("✅ Disease info loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading disease info: {e}")
    disease_info_db = {}


# ========== Routes ==========

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/chatbot")
def chatbot():
    if "username" not in session:
        return redirect(url_for("home"))
    return render_template("chatbot.html")

@app.route("/feedback")
def feedback():
    return render_template("feedback.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))

# ========== Signup/Login ==========

@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username")
    password = request.form.get("password")

    if not username or not password:
        return jsonify({"success": False, "message": "Please enter both username and password."})

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({"success": False, "message": "Username already exists."})

    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"success": True, "message": "Signup successful! You can now log in."})

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    user = User.query.filter_by(username=username, password=password).first()
    if user:
        session["username"] = username
        return jsonify({"success": True, "message": f"Welcome, {username}!"})
    else:
        return jsonify({"success": False, "message": "Invalid username or password."})

# ========== Launch Streamlit Bot ==========

def is_streamlit_running():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'streamlit' in ' '.join(cmdline) and any("8501" in arg for arg in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            continue
    return False

@app.route('/mental_health_bot')
def open_mental_health_bot():
    if not is_streamlit_running():
        try:
            subprocess.Popen(["streamlit", "run", "mentalbot.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info("Streamlit mentalbot launched.")
        except Exception as e:
            logging.error(f"Error launching Streamlit app: {e}")
    return redirect("http://localhost:8501", code=302)

# ========== Disease Chatbot ==========

def extract_symptoms(text, known_symptoms):
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return [sym for sym in known_symptoms if sym in cleaned]

@app.route("/chat", methods=["POST"])
def chat():
    if "username" not in session:
        return jsonify({"reply": "Please log in to use the chatbot."})

    user_input = request.json.get("message", "")
    valid_symptoms = extract_symptoms(user_input, all_symptoms)

    if not valid_symptoms:
        return jsonify({"reply": "😕 Sorry, I couldn't recognize any valid symptoms. Please rephrase or try different words."})

    input_vector = vectorizer.transform([" ".join(valid_symptoms)])
    predicted_disease = predictor_model.predict(input_vector)[0]

    prompt = f"""
You are a medical assistant chatbot. Based on the user's symptoms and predicted disease, give a **short response** with each point on a **new line**, using the exact format below.

Each point should be 1-2 lines max. Be clear and simple.

Format:
1. Disease: (1-line description + predicted disease name)
2. Symptoms: (detected symptoms from user)
3. Severity: (how serious the disease is)
4. Recommended Tests: (list tests)
5. Diet: (diet suggestions)
6. Lifestyle: (lifestyle or behavior advice)
7. Home Remedies: (remedy list)
8. Medicines: (common over-the-counter or prescription meds)

User Input: {user_input}
Detected Symptoms: {', '.join(valid_symptoms)}
Predicted Disease: {predicted_disease}
"""

    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return jsonify({"reply": response.text.strip()})
    except Exception as e:
        logging.error(f"Gemini API Error: {e}")
        return jsonify({"reply": "🚨 Sorry, I'm having trouble generating a response. Please try again later."})

# ========== Feedback Submission ==========

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    name = request.form.get("name")
    feedback = request.form.get("feedback")

    if not name or not feedback:
        return "Missing name or feedback", 400

    logging.info(f"Feedback from {name}: {feedback}")
    return redirect(url_for("home"))

# ========== Run App ==========

if __name__ == "__main__":
    app.run(debug=True)
