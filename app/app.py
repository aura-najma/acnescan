from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Validasi ekstensi
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model YOLO
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'best.pt')
model = YOLO(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        # 1. Ambil file dari form
        file = request.files.get('image')
        if not file or file.filename == '':
            print("[DEBUG] Tidak ada file yang dipilih.")
            return 'Tidak ada file yang dipilih'

        # 2. Validasi ekstensi file
        if not allowed_file(file.filename):
            print(f"[DEBUG] Format file tidak didukung: {file.filename}")
            return 'Format file tidak didukung (hanya .jpg, .jpeg, .png)'

        # 3. Simpan file ke folder uploads
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"[DEBUG] File disimpan di: {filepath}")

        # 4. Lakukan klasifikasi dengan model YOLOv8
        try:
            results = model(filepath)  # hasil = list
            print(f"[DEBUG] Raw result: {results}")
            preds = results[0].probs   # klasifikasi = .probs, bukan .boxes
        except Exception as e:
            print(f"[ERROR] Gagal melakukan prediksi: {e}")
            return 'Terjadi kesalahan saat memproses gambar'

        # 5. Ambil label dengan probabilitas tertinggi
        if preds is not None:
            class_id = preds.top1  # Index kelas dengan skor tertinggi
            predicted_label = model.names[class_id].lower()
            print(f"[DEBUG] Prediksi label: {predicted_label}")
        else:
            predicted_label = "tidak terdeteksi"
            print("[DEBUG] Tidak ada prediksi terdeteksi.")

        # 6. Redirect ke halaman hasil
        return redirect(url_for('result', image=filename, label=predicted_label))

    # Kalau GET, tampilkan halaman form upload
    return render_template('classify.html')



@app.route('/result')
def result():
    image = request.args.get('image')
    label = request.args.get('label')
    return render_template('result.html', image=image, label=label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
