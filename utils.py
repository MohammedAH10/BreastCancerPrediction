import os
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf # type: ignore

def load_model_from_hf(model_name):
    """Load TensorFlow model from Hugging Face Hub"""
    tf.get_logger().setLevel('ERROR')
    
    # Disable GPU if not needed (saves memory)
    tf.config.set_visible_devices([], 'GPU')
    
    # Download model file
    model_path = hf_hub_download(
        repo_id=model_name,
        filename="final_combined_model.keras",  # Replace with your actual filename if different
        use_auth_token=os.environ.get("HF_AUTH_TOKEN")
    )
    # Load TensorFlow model
    return load_model(model_path)

def preprocess_image(image):
    """Preprocess image for TensorFlow model"""
    # Convert to grayscale and resize
    image = image.convert('L').resize((224, 224))
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    # Add batch and channel dimensions
    return image_array[np.newaxis, ..., np.newaxis]

def predict_image(model, image):
    """Make prediction using TensorFlow model"""
    # Preprocess image
    processed_image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)[0][0]
    # Calculate confidence
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return prediction, confidence