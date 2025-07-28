from flask import Flask, render_template, request, redirect, url_for, make_response
import os
from werkzeug.utils import secure_filename
import numpy as np
import tifffile as tiff
from tensorflow.keras.models import load_model
import tensorflow as tf
from utils import normalize_image, preprocess_image, dice_coefficient, iou
from PIL import Image
import matplotlib.pyplot as plt
import time
import zipfile
import io
import json


app = Flask(__name__)


# Configure paths using os.path for cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_model.keras')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

ALLOWED_EXTENSIONS = {'tif', 'tiff'}


# Configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
ALLOWED_EXTENSIONS = {'tif', 'tiff'}


# Load model
try:
    model = load_model(MODEL_PATH, 
                      custom_objects={
                          'dice_coefficient': dice_coefficient,
                          'iou': iou
                      })
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded", 500
    
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        return "Invalid file type. Only .tif and .tiff files are allowed.", 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and preprocess image
        image = tiff.imread(filepath)  # Shape: (H, W, 12)
        norm_img = normalize_image(image)
        selected_img = preprocess_image(norm_img)

        # Make prediction
        prediction = model.predict(np.expand_dims(selected_img, axis=0))[0]
        binary_mask = (prediction > 0.5).astype(np.uint8).squeeze()
        
        # Calculate metrics (assuming 10m resolution per pixel)
        total_pixels = binary_mask.size
        water_pixels = np.sum(binary_mask)
        water_percentage = (water_pixels / total_pixels) * 100
        area_km2 = (water_pixels * 100) / 1_000_000  # Convert to km²

        # Define output paths
        rgb_path = os.path.join(app.config['STATIC_FOLDER'], 'rgb.png')
        mask_path = os.path.join(app.config['STATIC_FOLDER'], 'pred_mask.png')

        # Clean up existing files
        for path in [rgb_path, mask_path]:
            if os.path.exists(path):
                os.remove(path)

        # Save RGB image
        rgb = selected_img[:, :, :3]
        rgb_img = (rgb * 255).astype(np.uint8)
        Image.fromarray(rgb_img).save(rgb_path, format='PNG')


        # Save predicted mask
        mask_img = (binary_mask * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(mask_path, format='PNG')

        return render_template('result.html', 
                            timestamp=int(time.time()),
                            water_area=f"{area_km2:.1f} km²",
                            water_percent=f"{water_percentage:.1f}%")
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"An error occurred: {str(e)}", 500
    
@app.route('/export')
def export_data():
    try:
        # Create in-memory zip file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add images
            for filename in ['rgb.png', 'pred_mask.png']:
                filepath = os.path.join(app.config['STATIC_FOLDER'], filename)
                if os.path.exists(filepath):
                    zip_file.write(filepath, filename)
            
            # Add analysis metadata
            analysis_data = {
                "water_area": request.args.get('water_area', 'N/A'),
                "water_percent": request.args.get('water_percent', 'N/A'),
            }
            zip_file.writestr('analysis.json', json.dumps(analysis_data, indent=2))
        
        # Prepare response
        zip_buffer.seek(0)
        response = make_response(zip_buffer.read())
        response.headers['Content-Type'] = 'application/zip'
        response.headers['Content-Disposition'] = 'attachment; filename=water_analysis.zip'
        return response
    
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True)