from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model YOLO
# model = YOLO('app/model/best.pt')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Prediksi menggunakan YOLO
            results = model(filepath)
            labels = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    labels.append(class_name)

            # Ambil label paling umum atau pertama
            predicted_label = labels[0] if labels else "Tidak terdeteksi"

            return redirect(url_for('result', image=filename, label=predicted_label))
    return render_template('classify.html')

@app.route('/result')
def result():
    image = request.args.get('image', None)
    label = request.args.get('label', None)
    return render_template('result.html', image=image, label=label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
