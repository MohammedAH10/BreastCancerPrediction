# app.py
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # ignore

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '443abfe9ce5b627e541aa8956523246fb3f0ae0fcc8b70ddfd95821905033c6e'

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_models" / "final_combined_model.keras"
DB_PATH = BASE_DIR / "database" / "predictions.db"
UPLOAD_DIR = BASE_DIR / "uploads"

# Create required directories
for path in [UPLOAD_DIR, DB_PATH.parent, MODEL_PATH.parent]:
    path.mkdir(parents=True, exist_ok=True)

print(f"Model path: {MODEL_PATH}")
print(f"Database path: {DB_PATH}")
print(f"Upload directory: {UPLOAD_DIR}")

# ===== Database Setup =====
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                prediction REAL,
                confidence REAL,
                image_path TEXT
            )
        ''')
        conn.commit()

# Initialize database
init_db()

# ===== Load ML Model =====
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

# ===== Helper Functions =====
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    # Convert to grayscale and resize
    image = image.convert('L').resize((224, 224))
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    # Add batch and channel dimensions
    return image_array[np.newaxis, ..., np.newaxis]

def predict_image(image_data: np.ndarray) -> tuple:
    """Make prediction using loaded model"""
    if model is None:
        raise RuntimeError("Model not loaded")
    
    prediction = model.predict(image_data, verbose=0)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return prediction, float(confidence)

def save_to_db(prediction_data: dict):
    """Save prediction to SQLite database"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (id, timestamp, prediction, confidence, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            prediction_data['id'],
            prediction_data['timestamp'],
            prediction_data['prediction'],
            prediction_data['confidence'],
            prediction_data['image_path']
        ))
        conn.commit()

# ===== Routes =====
@app.route('/')
def home():
    """Render home page with upload form"""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'image' not in request.files:
        flash('No image uploaded', 'error')
        return redirect(url_for('home'))
    
    image = request.files['image']
    if image.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('home'))
    
    try:
        # Generate unique ID
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Save uploaded file
        filename = f"{prediction_id}_{image.filename}"
        file_path = UPLOAD_DIR / filename
        image.save(file_path)
        
        # Preprocess image
        img = Image.open(file_path)
        processed_image = preprocess_image(img)
        
        # Make prediction
        prediction, confidence = predict_image(processed_image)
        result = "Malignant" if prediction > 0.5 else "Benign"
        
        # Prepare prediction data
        prediction_data = {
            "id": prediction_id,
            "timestamp": timestamp.isoformat(),
            "prediction": float(prediction),
            "confidence": confidence,
            "image_path": str(file_path)
        }
        
        # Save to database
        save_to_db(prediction_data)
        
        # Prepare response
        return render_template('result.html', 
                               result=result,
                               confidence=f"{confidence*100:.2f}%",
                               prediction_id=prediction_id,
                               image_url=f"/uploads/{filename}",
                               now=timestamp)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        flash(f'Prediction error: {str(e)}', 'error')
        return redirect(url_for('home'))
@app.route('/history')
def history():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, prediction, confidence, image_path
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()
    return render_template("history.html", rows=rows)

if __name__ == '__main__':
    app.run(debug=True)