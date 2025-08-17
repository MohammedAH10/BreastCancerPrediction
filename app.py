import os
import logging
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image, UnidentifiedImageError
import numpy as np
from utils import load_model_from_hf, predict_image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# ===== Configuration =====
HF_MODEL_NAME = "MohammedAH/BreastCancerPrediction"  # Replace with your model name
DB_PATH = "database/predictions.db"
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("database", exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

# ===== Database Setup =====
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                age INTEGER,
                tumor_size REAL,
                prediction REAL,
                confidence REAL,
                image_path TEXT,
                source TEXT
            )
        ''')
        conn.commit()

# Initialize database
init_db()

# ===== Load Hugging Face Model =====
try:
    model = load_model_from_hf(HF_MODEL_NAME)
    logging.info("Successfully loaded TensorFlow model from Hugging Face Hub")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    model = None

# ===== Helper Functions =====
def save_to_db(prediction_data: dict):
    """Save prediction to SQLite database"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (id, timestamp, age, tumor_size, prediction, confidence, image_path, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_data['id'],
            prediction_data['timestamp'],
            prediction_data['age'],
            prediction_data['tumor_size'],
            prediction_data['prediction'],
            prediction_data['confidence'],
            prediction_data['image_path'],
            prediction_data['source']
        ))
        conn.commit()

# ===== Routes =====
@app.route('/')
def home():
    """Render home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return prediction result"""
    # Get form data
    age = request.form.get('age')
    tumor_size = request.form.get('tumor_size')
    image_file = request.files.get('image')
    
    # Validate inputs
    errors = []
    if not age or not age.isdigit():
        errors.append("Please enter a valid age")
    if not tumor_size or not tumor_size.replace('.', '', 1).isdigit():
        errors.append("Please enter a valid tumor size")
    if not image_file or image_file.filename == '':
        errors.append("Please select an image file")
    
    if errors:
        for error in errors:
            flash(error, 'danger')
        return redirect(url_for('home'))
    
    # Generate unique ID for this prediction
    prediction_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    try:
        # Convert inputs to proper types
        age = int(age)
        tumor_size = float(tumor_size)
        
        # Save uploaded file
        file_ext = os.path.splitext(image_file.filename)[1]
        file_path = Path(UPLOAD_DIR) / f"{prediction_id}{file_ext}"
        image_file.save(file_path)
        
        # Make prediction
        if model is None:
            raise Exception("Model not loaded")
        
        img = Image.open(file_path)
        prediction, confidence = predict_image(model, img)
        result = "Malignant" if prediction > 0.5 else "Benign"
        
        # Prepare prediction data
        prediction_data = {
            "id": prediction_id,
            "timestamp": timestamp,
            "age": age,
            "tumor_size": tumor_size,
            "prediction": float(prediction),
            "confidence": confidence,
            "image_path": str(file_path),
            "source": "web_form"
        }
        
        # Save to database
        save_to_db(prediction_data)
        
        # Prepare response
        return render_template('result.html',
            result=result,
            confidence=f"{confidence*100:.2f}%",
            age=age,
            tumor_size=tumor_size,
            image_url=f"/{file_path}",
            prediction_id=prediction_id
        )
    
    except UnidentifiedImageError:
        flash("Invalid image file format. Please upload a valid image (JPEG, PNG, etc.)", 'danger')
        return redirect(url_for('home'))
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        flash(f"Prediction error: {str(e)}", 'danger')
        return redirect(url_for('home'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
