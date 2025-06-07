import logging
from flask import Flask, render_template, request, redirect, url_for, current_app
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2
import base64
from io import BytesIO
from PIL import Image

# Define the Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set logging to debug mode
app.logger.setLevel(logging.DEBUG)  # Ensure Flask logs at the DEBUG level

# Configuration for file upload
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# YOLO model loading
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'best.pt')
model = YOLO(MODEL_PATH)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image_from_data_url(data_url):
    # Decode the base64 image
    image_data = base64.b64decode(data_url.split(',')[1])
    image = Image.open(BytesIO(image_data))

    # Debugging: log image size
    app.logger.debug(f"[DEBUG] Image size (bytes): {len(image_data)}")

    # Save the image to the server
    filename = 'captured_image.png'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)
    return filepath

def predict_image(image_path):
    try:
        results = model(image_path)
        preds = results[0].probs
        if preds is not None:
            class_id = preds.top1  # Class index with the highest probability
            predicted_label = model.names[class_id].lower()
            app.logger.debug(f"[DEBUG] Predicted label: {predicted_label}")
            return predicted_label
        else:
            app.logger.debug("[DEBUG] No predictions detected.")
            return "tidak terdeteksi"
    except Exception as e:
        app.logger.error(f"[ERROR] Prediction failed: {e}")
        return "tidak terdeteksi"

def detect_face(image_path):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            app.logger.debug("[DEBUG] No face detected in the image.")
            return False  # No face detected
        app.logger.debug(f"[DEBUG] Faces detected: {len(faces)}")
        return True  # Face detected
    except Exception as e:
        app.logger.error(f"[ERROR] Face detection failed: {e}")
        return False

# Routes
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    filename = None  # Initialize filename to avoid UnboundLocalError

    if request.method == 'POST':
        # Handle regular file upload
        file = request.files.get('image')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            app.logger.debug(f"[DEBUG] Uploaded file size (bytes): {len(file.read())}")
            file.seek(0)
        else:
            # Handle image from camera (base64 data)
            image_data = request.form.get('image')
            if image_data:
                filename = 'captured_image.png'
                filepath = save_image_from_data_url(image_data)
            else:
                return "No image data received", 400

        # Step 1: Classify the image first
        predicted_label = predict_image(filepath)

        # Step 2: If the label is "IGA0", check for face detection
        if predicted_label in ["iga0", "iga1"]:
            app.logger.debug(f"[DEBUG] {predicted_label.upper()} detected, checking for face...")
            if not detect_face(filepath):
                app.logger.debug("[DEBUG] No face detected.")
                return redirect(url_for('result', image=filename, label='tidak-terdeteksi'))
       
        # Step 3: Redirect to the result page with the predicted label
        return redirect(url_for('result', image=filename, label=predicted_label))

    return render_template('classify.html')  

@app.route('/result')
def result():
    image = request.args.get('image')
    label = request.args.get('label')
    return render_template('result.html', image=image, label=label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
