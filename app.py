# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import bcrypt
from datetime import datetime, timedelta
import joblib
from functools import wraps
import os
import re
import pandas as pd
import nltk
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pickle
from PIL import Image
import pytesseract
import jwt
import traceback
import csv
from io import StringIO
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your-secret-key-here-change-in-production'
JWT_SECRET = "jwt-secret-key-123-change-in-production"
JWT_ALGO = "HS256"
JWT_EXP_MINUTES = 60

CORS(app)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '1975',
    'database': 'job_fraud_db'
}

# ML Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize NLTK
nltk.download('stopwords', quiet=True)
stopwords = set(nltk.corpus.stopwords.words('english'))

# Initialize SpaCy
try:
    nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
except:
    # If SpaCy model not available, use a fallback
    nlp = None

# Initialize Tesseract (if available)
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except:
    pytesseract = None

# ============================================
# TEXT CLEANING FUNCTIONS
# ============================================

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text)
    
    # Remove URLs & websites
    text = re.sub(r'http\S+|www\S+', ' ', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove phone numbers
    text = re.sub(r'\+?\d[\d -]{8,12}\d', ' ', text)
    
    # Remove emojis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    
    # Keep only alphabets, numbers, spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def lemmatize(text):
    """Lemmatize text using SpaCy"""
    if nlp and text:
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if token.text not in stopwords])
    return text

# ============================================
# ML MODEL LOADING AND MANAGEMENT
# ============================================

class ModelManager:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all ML models"""
        try:
            # Load TF-IDF + Logistic Regression/NB models
            tfidf_path = os.path.join(MODELS_DIR, 'tfidf.pkl')
            log_model_path = os.path.join(MODELS_DIR, 'log_model.pkl')
            nb_model_path = os.path.join(MODELS_DIR, 'nb_model.pkl')
            
            if os.path.exists(tfidf_path):
                self.vectorizers['tfidf'] = joblib.load(tfidf_path)
            
            if os.path.exists(log_model_path):
                self.models['logistic'] = joblib.load(log_model_path)
            
            if os.path.exists(nb_model_path):
                self.models['naive_bayes'] = joblib.load(nb_model_path)
            
            # Load TensorFlow models
            cnn_model_path = os.path.join(MODELS_DIR, 'cnn_model.h5')
            rnn_model_path = os.path.join(MODELS_DIR, 'rnn_model.h5')
            
            if os.path.exists(cnn_model_path):
                self.models['cnn'] = load_model(cnn_model_path)
            
            if os.path.exists(rnn_model_path):
                self.models['rnn'] = load_model(rnn_model_path)
            
            # Load DistilBERT model
            distilbert_path = os.path.join(MODELS_DIR, 'distilbert_model')
            if os.path.exists(distilbert_path):
                self.models['distilbert'] = DistilBertForSequenceClassification.from_pretrained(distilbert_path)
                self.vectorizers['distilbert_tokenizer'] = DistilBertTokenizerFast.from_pretrained(distilbert_path)
            
            print("✓ Models loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_tfidf(self, text, model_type='logistic'):
        """Predict using TF-IDF based models"""
        if 'tfidf' not in self.vectorizers or model_type not in self.models:
            return None, 50
        
        cleaned = clean_text(text)
        vector = self.vectorizers['tfidf'].transform([cleaned])
        
        if model_type == 'logistic':
            prediction = self.models['logistic'].predict(vector)[0]
        elif model_type == 'naive_bayes':
            prediction = self.models['naive_bayes'].predict(vector)[0]
        else:
            return None, 50
        
        return prediction
    
    def predict_deep_learning(self, text, model_type='cnn'):
        """Predict using deep learning models"""
        if model_type not in self.models or model_type not in ['cnn', 'rnn']:
            return None, 50
        
        # Preprocess text for DL models
        cleaned = clean_text(text)
        lemmatized = lemmatize(cleaned)
        
        # Tokenize and pad (using simple tokenizer for demo)
        # In production, use the same tokenizer used during training
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts([lemmatized])
        sequences = tokenizer.texts_to_sequences([lemmatized])
        padded = pad_sequences(sequences, maxlen=200, padding='post')
        
        prediction = self.models[model_type].predict(padded)[0][0]
        return 1 if prediction > 0.5 else 0
    
    def predict_distilbert(self, text):
        """Predict using DistilBERT"""
        if 'distilbert' not in self.models or 'distilbert_tokenizer' not in self.vectorizers:
            return None, 50
        
        try:
            tokenizer = self.vectorizers['distilbert_tokenizer']
            model = self.models['distilbert']
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            model.eval()
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = probs.argmax().item()
            confidence = probs[0][pred].item() * 100
            
            return pred, confidence
            
        except Exception as e:
            print(f"DistilBERT prediction error: {e}")
            return None, 50
    
    def ensemble_predict(self, text):
        """Use ensemble of models for prediction"""
        predictions = []
        confidences = []
        
        # Try TF-IDF models
        try:
            log_pred = self.predict_tfidf(text, 'logistic')
            if log_pred is not None:
                predictions.append(log_pred)
                confidences.append(70)  # Default confidence
        except:
            pass
        
        try:
            nb_pred = self.predict_tfidf(text, 'naive_bayes')
            if nb_pred is not None:
                predictions.append(nb_pred)
                confidences.append(65)  # Default confidence
        except:
            pass
        
        # Try DistilBERT
        try:
            distil_pred, distil_conf = self.predict_distilbert(text)
            if distil_pred is not None:
                predictions.append(distil_pred)
                confidences.append(distil_conf)
        except:
            pass
        
        # Take majority vote
        if predictions:
            # Count predictions
            fake_count = sum(1 for p in predictions if p == 1)
            real_count = sum(1 for p in predictions if p == 0)
            
            if fake_count > real_count:
                final_pred = 1
                final_conf = np.mean([c for p, c in zip(predictions, confidences) if p == 1])
            else:
                final_pred = 0
                final_conf = np.mean([c for p, c in zip(predictions, confidences) if p == 0])
            
            return "Fake" if final_pred == 1 else "Real", min(99, max(50, final_conf))
        
        return "Uncertain", 50

model_manager = ModelManager()

# ============================================
# DATABASE FUNCTIONS
# ============================================

def get_db_connection():
    """Create database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def create_tables():
    """Create database tables if they don't exist"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(100) UNIQUE,
                password VARCHAR(255),
                role VARCHAR(20) DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL,
                status VARCHAR(20) DEFAULT 'active'
            )
        """)
        
        # Create job_checks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_checks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                job_title VARCHAR(255),
                prediction VARCHAR(20),
                confidence DECIMAL(5,2),
                feedback_status VARCHAR(20) DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Create image_checks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_checks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                image_name VARCHAR(255),
                prediction VARCHAR(20),
                confidence DECIMAL(5,2),
                feedback_status VARCHAR(20) DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Create user_stats table (optional, for caching)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id INT PRIMARY KEY,
                total_scans INT DEFAULT 0,
                real_jobs INT DEFAULT 0,
                fake_jobs INT DEFAULT 0,
                avg_confidence DECIMAL(5,2) DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✓ Database tables created/verified")
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

# Initialize database tables
create_tables()

# ============================================
# AUTHENTICATION FUNCTIONS
# ============================================

def create_jwt(user):
    """Create JWT token for user"""
    payload = {
        "user_id": user["id"],
        "email": user["email"],
        "role": user["role"],
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXP_MINUTES)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def verify_jwt(token):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def login_required(f):
    """Decorator for requiring login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth = request.headers.get("Authorization")
        
        if not auth:
            return jsonify({"success": False, "message": "Token missing"}), 401
        
        try:
            token = auth.replace("Bearer ", "")
            payload = verify_jwt(token)
            
            if not payload:
                return jsonify({"success": False, "message": "Invalid or expired token"}), 401
            
            # Add user info to request context
            request.user_id = payload["user_id"]
            request.user_role = payload["role"]
            
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator for requiring admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(request, 'user_role') or request.user_role != 'admin':
            return jsonify({"success": False, "message": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated_function

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_text_from_image(image_file):
    """Extract text from image using OCR"""
    if not pytesseract:
        return None
    
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return None

def analyze_job_text(text):
    """Analyze job text using ensemble of models"""
    if not text or len(text.strip()) < 20:
        return "Uncertain", 50, {"fake": 0, "real": 0}
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Get prediction from ensemble
    prediction, confidence = model_manager.ensemble_predict(cleaned_text)
    
    # Count indicators (simple keyword matching for demo)
    fake_indicators = ["upfront payment", "registration fee", "work from home", 
                      "no experience needed", "earn money fast", "get rich quick"]
    real_indicators = ["competitive salary", "benefits package", "health insurance",
                      "401k", "professional development", "required qualifications"]
    
    fake_count = sum(1 for indicator in fake_indicators if indicator in text.lower())
    real_count = sum(1 for indicator in real_indicators if indicator in text.lower())
    
    return prediction, confidence, {"fake": fake_count, "real": real_count}

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard"""
    return render_template('dashboard.html')

@app.route('/checkjob')
def check_job():
    """Serve the job checking page"""
    return render_template('checkjob.html')

@app.route('/admin')
def admin_page():
    """Serve the admin page"""
    return render_template('admin.html')

# ============================================
# AUTHENTICATION ROUTES
# ============================================

@app.route('/api/signup', methods=['POST'])
def signup():
    """User registration"""
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if not all([name, email, password]):
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        
        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # Check if email exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Email already registered'}), 400
        
        # Insert new user
        cursor.execute(
            "INSERT INTO users (name, email, password, role) VALUES (%s, %s, %s, 'user')",
            (name, email, hashed_password.decode('utf-8'))
        )
        user_id = cursor.lastrowid
        
        # Create user stats entry
        cursor.execute(
            "INSERT INTO user_stats (user_id) VALUES (%s)",
            (user_id,)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully! Please sign in.',
            'user_id': user_id
        })
        
    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({'success': False, 'message': 'An error occurred during signup'}), 500

@app.route('/api/signin', methods=['POST'])
def signin():
    """User login"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # Get user
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        
        if not user:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
        
        # Verify password
        if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            # Update last login
            cursor.execute(
                "UPDATE users SET last_login = NOW() WHERE id = %s",
                (user['id'],)
            )
            conn.commit()
            
            # Remove password from response
            user.pop('password')
            
            # Create JWT token
            token = create_jwt(user)
            
            cursor.close()
            conn.close()
            
            return jsonify({
                "success": True,
                "token": token,
                "user": {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                    "role": user["role"]
                }
            })
        else:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
            
    except Exception as e:
        print(f"Signin error: {e}")
        return jsonify({'success': False, 'message': 'An error occurred during signin'}), 500

@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    """Password reset"""
    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")
        confirm_password = data.get("confirm_password")

        if not all([email, password, confirm_password]):
            return jsonify({"success": False, "message": "All fields are required"}), 400

        if password != confirm_password:
            return jsonify({"success": False, "message": "Passwords do not match"}), 400

        if len(password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if not user:
            cursor.close()
            conn.close()
            return jsonify({"success": False, "message": "Email not registered"}), 404

        # Hash new password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Update password
        cursor.execute(
            "UPDATE users SET password=%s WHERE email=%s",
            (hashed_password.decode('utf-8'), email)
        )

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Password reset successful. Please login."
        })

    except Exception as e:
        print(f"Password reset error: {e}")
        return jsonify({"success": False, "message": "An error occurred"}), 500

# ============================================
# JOB PREDICTION ROUTES
# ============================================

@app.route('/api/predict/text', methods=['POST'])
@login_required
def predict_text():
    """Predict job legitimacy from text"""
    try:
        data = request.json
        user_id = request.user_id
        
        # Combine all text fields
        text_parts = [
            data.get("title", ""),
            data.get("company_profile", ""),
            data.get("description", ""),
            data.get("requirements", ""),
            data.get("benefits", "")
        ]
        
        full_text = " ".join([str(part) for part in text_parts])
        
        if not full_text.strip():
            return jsonify({"success": False, "message": "No text provided"}), 400
        
        # Analyze text
        prediction, confidence, indicators = analyze_job_text(full_text)
        
        # Save to database
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO job_checks (user_id, job_title, prediction, confidence)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, data.get('title', 'Untitled Job'), prediction, confidence)
            )
            
            # Update user stats
            cursor.execute(
                """
                INSERT INTO user_stats (user_id, total_scans, real_jobs, fake_jobs, avg_confidence)
                VALUES (%s, 1, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                total_scans = total_scans + 1,
                real_jobs = real_jobs + %s,
                fake_jobs = fake_jobs + %s,
                avg_confidence = (avg_confidence * (total_scans - 1) + %s) / total_scans
                """,
                (
                    user_id,
                    1 if prediction == "Real" else 0,
                    1 if prediction == "Fake" else 0,
                    confidence,
                    1 if prediction == "Real" else 0,
                    1 if prediction == "Fake" else 0,
                    confidence
                )
            )
            
            record_id = cursor.lastrowid
            conn.commit()
            cursor.close()
            conn.close()
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "indicators": indicators,
            "record_id": record_id if 'record_id' in locals() else None
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"success": False, "message": "Prediction failed"}), 500

@app.route('/api/predict/image', methods=['POST'])
@login_required
def predict_image():
    """Predict job legitimacy from image"""
    try:
        user_id = request.user_id
        
        if 'image' not in request.files:
            return jsonify({"success": False, "message": "No image uploaded"}), 400
        
        image_file = request.files['image']
        image_name = image_file.filename
        
        # Extract text from image
        extracted_text = extract_text_from_image(image_file)
        
        if not extracted_text or len(extracted_text.strip()) < 20:
            prediction, confidence, indicators = "Uncertain", 50, {"fake": 0, "real": 0}
        else:
            # Analyze extracted text
            prediction, confidence, indicators = analyze_job_text(extracted_text)
        
        # Save to database
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO image_checks (user_id, image_name, prediction, confidence)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, image_name, prediction, confidence)
            )
            
            # Update user stats
            cursor.execute(
                """
                INSERT INTO user_stats (user_id, total_scans, real_jobs, fake_jobs, avg_confidence)
                VALUES (%s, 1, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                total_scans = total_scans + 1,
                real_jobs = real_jobs + %s,
                fake_jobs = fake_jobs + %s,
                avg_confidence = (avg_confidence * (total_scans - 1) + %s) / total_scans
                """,
                (
                    user_id,
                    1 if prediction == "Real" else 0,
                    1 if prediction == "Fake" else 0,
                    confidence,
                    1 if prediction == "Real" else 0,
                    1 if prediction == "Fake" else 0,
                    confidence
                )
            )
            
            record_id = cursor.lastrowid
            conn.commit()
            cursor.close()
            conn.close()
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "indicators": indicators,
            "extracted_text": extracted_text[:500] if extracted_text else "",
            "record_id": record_id if 'record_id' in locals() else None
        })
        
    except Exception as e:
        print(f"Image prediction error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": "Image prediction failed"}), 500

# ============================================
# HISTORY & STATISTICS ROUTES
# ============================================

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """Get user's scan history"""
    try:
        user_id = request.user_id
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get text predictions
        cursor.execute("""
            SELECT id, job_title AS title, prediction, confidence, created_at,
                   'text' AS source
            FROM job_checks
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 50
        """, (user_id,))
        text_history = cursor.fetchall()
        
        # Get image predictions
        cursor.execute("""
            SELECT id, image_name AS title, prediction, confidence, created_at,
                   'image' AS source
            FROM image_checks
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 50
        """, (user_id,))
        image_history = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Combine and sort
        full_history = text_history + image_history
        full_history.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            'success': True,
            'count': len(full_history),
            'history': full_history
        })
        
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch history'}), 500

@app.route('/api/stats/user', methods=['GET'])
@login_required
def get_user_stats():
    """Get user statistics"""
    try:
        user_id = request.user_id
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get user stats
        cursor.execute("""
            SELECT total_scans, real_jobs, fake_jobs, avg_confidence
            FROM user_stats
            WHERE user_id = %s
        """, (user_id,))
        stats = cursor.fetchone()
        
        if not stats:
            stats = {
                'total_scans': 0,
                'real_jobs': 0,
                'fake_jobs': 0,
                'avg_confidence': 0
            }
        
        # Get recent scans count
        cursor.execute("""
            SELECT COUNT(*) as recent_scans
            FROM (
                SELECT created_at FROM job_checks WHERE user_id = %s
                UNION ALL
                SELECT created_at FROM image_checks WHERE user_id = %s
            ) as all_scans
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        """, (user_id, user_id))
        recent = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                **stats,
                'recent_scans': recent['recent_scans'] if recent else 0
            }
        })
        
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch stats'}), 500

# ============================================
# ADMIN ROUTES
# ============================================

@app.route('/api/admin/stats', methods=['GET'])
@login_required
@admin_required
def get_admin_stats():
    """Get admin statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get total counts
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT u.id) as total_users,
                COUNT(DISTINCT jc.id) + COUNT(DISTINCT ic.id) as total_scans,
                SUM(CASE WHEN jc.prediction = 'Real' THEN 1 ELSE 0 END) + 
                SUM(CASE WHEN ic.prediction = 'Real' THEN 1 ELSE 0 END) as real_jobs,
                SUM(CASE WHEN jc.prediction = 'Fake' THEN 1 ELSE 0 END) + 
                SUM(CASE WHEN ic.prediction = 'Fake' THEN 1 ELSE 0 END) as fake_jobs
            FROM users u
            LEFT JOIN job_checks jc ON u.id = jc.user_id
            LEFT JOIN image_checks ic ON u.id = ic.user_id
        """)
        totals = cursor.fetchone()
        
        # Get daily stats for last 7 days
        cursor.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as count
            FROM (
                SELECT created_at FROM job_checks
                UNION ALL
                SELECT created_at FROM image_checks
            ) as all_scans
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            GROUP BY DATE(created_at)
            ORDER BY date
        """)
        daily_stats = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_users': totals['total_users'] or 0,
                'total_scans': totals['total_scans'] or 0,
                'real_jobs': totals['real_jobs'] or 0,
                'fake_jobs': totals['fake_jobs'] or 0,
                'daily_stats': daily_stats
            }
        })
        
    except Exception as e:
        print(f"Admin stats error: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch admin stats'}), 500

@app.route('/api/admin/users', methods=['GET'])
@login_required
@admin_required
def get_admin_users():
    """Get all users for admin"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                u.id, u.name, u.email, u.role, u.created_at, u.last_login, u.status,
                COALESCE(us.total_scans, 0) as total_scans,
                COALESCE(us.real_jobs, 0) as real_jobs,
                COALESCE(us.fake_jobs, 0) as fake_jobs
            FROM users u
            LEFT JOIN user_stats us ON u.id = us.user_id
            ORDER BY u.created_at DESC
        """)
        users = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'users': users
        })
        
    except Exception as e:
        print(f"Admin users error: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch users'}), 500

@app.route('/api/admin/scans', methods=['GET'])
@login_required
@admin_required
def get_admin_scans():
    """Get all scans for admin"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get text scans
        cursor.execute("""
            SELECT 
                jc.id, jc.job_title as title, jc.prediction, jc.confidence, 
                jc.created_at, 'text' as source,
                u.name as user_name, u.email as user_email
            FROM job_checks jc
            JOIN users u ON jc.user_id = u.id
            ORDER BY jc.created_at DESC
            LIMIT 100
        """)
        text_scans = cursor.fetchall()
        
        # Get image scans
        cursor.execute("""
            SELECT 
                ic.id, ic.image_name as title, ic.prediction, ic.confidence, 
                ic.created_at, 'image' as source,
                u.name as user_name, u.email as user_email
            FROM image_checks ic
            JOIN users u ON ic.user_id = u.id
            ORDER BY ic.created_at DESC
            LIMIT 100
        """)
        image_scans = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Combine scans
        all_scans = text_scans + image_scans
        all_scans.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            'success': True,
            'scans': all_scans[:100]  # Limit to 100
        })
        
    except Exception as e:
        print(f"Admin scans error: {e}")
        return jsonify({'success': False, 'message': 'Failed to fetch scans'}), 500

# ============================================
# FEEDBACK ROUTES
# ============================================

@app.route('/api/feedback', methods=['POST'])
@login_required
def submit_feedback():
    """Submit feedback on a prediction"""
    try:
        data = request.json
        user_id = request.user_id
        record_id = data.get('record_id')
        feedback = data.get('feedback')  # 'correct' or 'incorrect'
        source = data.get('source')  # 'text' or 'image'
        
        if not all([record_id, feedback, source]):
            return jsonify({"success": False, "message": "Missing required fields"}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if source == 'text':
            cursor.execute(
                "UPDATE job_checks SET feedback_status = %s WHERE id = %s AND user_id = %s",
                (feedback, record_id, user_id)
            )
        elif source == 'image':
            cursor.execute(
                "UPDATE image_checks SET feedback_status = %s WHERE id = %s AND user_id = %s",
                (feedback, record_id, user_id)
            )
        else:
            cursor.close()
            conn.close()
            return jsonify({"success": False, "message": "Invalid source"}), 400
        
        affected_rows = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()
        
        if affected_rows == 0:
            return jsonify({"success": False, "message": "Record not found or access denied"}), 404
        
        return jsonify({"success": True, "message": "Feedback submitted"})
        
    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({"success": False, "message": "Failed to submit feedback"}), 500

# ============================================
# MODEL MANAGEMENT ROUTES (ADMIN ONLY)
# ============================================

@app.route('/api/admin/models/retrain', methods=['POST'])
@login_required
@admin_required
def retrain_models():
    """Retrain ML models with new data"""
    try:
        # In production, this would trigger model retraining
        # For now, just reload existing models
        
        success = model_manager.load_all_models()
        
        if success:
            return jsonify({
                "success": True,
                "message": "Models reloaded successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to reload models"
            }), 500
        
    except Exception as e:
        print(f"Model retrain error: {e}")
        return jsonify({"success": False, "message": "Failed to retrain models"}), 500

# ============================================
# EXPORT ROUTES
# ============================================

@app.route('/api/export/history', methods=['GET'])
@login_required
def export_history():
    """Export user history as CSV"""
    try:
        user_id = request.user_id
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get all user scans
        cursor.execute("""
            SELECT 
                'text' as type,
                job_title as title,
                prediction,
                confidence,
                created_at
            FROM job_checks
            WHERE user_id = %s
            UNION ALL
            SELECT 
                'image' as type,
                image_name as title,
                prediction,
                confidence,
                created_at
            FROM image_checks
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id, user_id))
        
        scans = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Create CSV
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Type', 'Title', 'Prediction', 'Confidence', 'Date'])
        
        # Write data
        for scan in scans:
            writer.writerow([
                scan['type'].upper(),
                scan['title'],
                scan['prediction'],
                scan['confidence'],
                scan['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        # Prepare response
        output.seek(0)
        
        return send_file(
            StringIO(output.getvalue()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='jobguard_history.csv'
        )
        
    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({"success": False, "message": "Failed to export history"}), 500

# ============================================
# HEALTH CHECK & UTILITY ROUTES
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if get_db_connection() else "disconnected",
        "models_loaded": len(model_manager.models) > 0
    })

@app.route('/api/test/predict', methods=['POST'])
def test_predict():
    """Test prediction endpoint"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"success": False, "message": "No text provided"}), 400
        
        prediction, confidence, indicators = analyze_job_text(text)
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "indicators": indicators
        })
        
    except Exception as e:
        print(f"Test predict error: {e}")
        return jsonify({"success": False, "message": "Test prediction failed"}), 500

# ============================================
# STATIC FILE SERVING
# ============================================

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_file(os.path.join('static', filename))

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "message": "Internal server error"}), 500

# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Create sample HTML files if they don't exist
    html_files = {
        'index.html': """
<!DOCTYPE html>
<html>
<head>
    <title>JobGuard Pro - AI Job Scam Detection</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>JobGuard Pro</h1>
    <p>AI-Powered Job Scam Detection System</p>
    <div>
        <a href="/dashboard">Go to Dashboard</a> | 
        <a href="/checkjob">Check a Job</a> | 
        <a href="/admin">Admin Panel</a>
    </div>
</body>
</html>
        """,
        'dashboard.html': """
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard - JobGuard Pro</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>User Dashboard</h1>
    <p>Welcome to JobGuard Pro Dashboard</p>
    <div id="app">
        <!-- React/Vue app would be loaded here -->
        <p>Loading dashboard...</p>
    </div>
    <script src="/static/js/app.js"></script>
</body>
</html>
        """
    }
    
    for filename, content in html_files.items():
        filepath = os.path.join('templates', filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(content)
    
    print("=" * 60)
    print("JobGuard Pro - AI Job Scam Detection System")
    print("=" * 60)
    print(f"Server running on: http://localhost:5000")
    print(f"API Base URL: http://localhost:5000/api")
    print(f"Models loaded: {len(model_manager.models)}")
    print("Available endpoints:")
    print("  GET  /                    - Main page")
    print("  GET  /dashboard           - User dashboard")
    print("  GET  /checkjob           - Job checking interface")
    print("  GET  /admin              - Admin panel")
    print("  POST /api/signin         - User login")
    print("  POST /api/signup         - User registration")
    print("  POST /api/predict/text   - Predict from text")
    print("  POST /api/predict/image  - Predict from image")
    print("  GET  /api/history        - Get user history")
    print("  GET  /api/stats/user     - Get user statistics")
    print("  GET  /api/health         - Health check")
    print("=" * 60)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
