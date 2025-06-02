from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os, time
from ultralytics import YOLO
import cv2

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = 't0pS3cr3tK3y@AcneScan2025!'

# Validasi ekstensi file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk memuat model dengan path dinamis dan mengukur waktu loading
# def load_model_with_time(model_folder):
#   model_path = os.path.join(os.path.dirname(__file__), 'model', model_folder, 'best.pt')
#   start_time = time.time()  # Catat waktu mulai loading
#   model = YOLO(model_path)  # Muat model
#   end_time = time.time()  # Catat waktu selesai loading
#   load_time = end_time - start_time  # Hitung waktu yang dibutuhkan
#  print(f"[DEBUG] Waktu loading model {model_path}: {load_time:.4f} detik")
#   return model, load_time

# Inisialisasi dictionary untuk menyimpan waktu loading model
# model_loading_times = {}

# Daftar folder model yang ingin diload
# models_folders = ['Large_384', 'Medium_384', 'Nano_512', 'Nano_384', 'Small_224', 'XL_512']

# Loop untuk memuat setiap model dan menyimpan waktu loadingnya
# for folder in models_folders:
#    model, load_time = load_model_with_time(folder)
#    model_loading_times[folder] = load_time
# Load model YOLO
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'best.pt')
model = YOLO(MODEL_PATH)
# Fungsi untuk deteksi wajah
def detect_face(image_path):
    # Load model deteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load gambar dan konversi ke grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Jika tidak ada wajah yang terdeteksi
    if len(faces) == 0:
        return False  # Gambar tidak valid untuk analisis jerawat
    return True  # Wajah terdeteksi

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sign-up')
def sign_up():
    
    return render_template('register.html')  # Render the sign-up page

@app.route('/sign-in')
def sign_in():
    return render_template('login.html')
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if 'terms_accepted' not in session or not session['terms_accepted']:
        return redirect(url_for('home'))  # Jika belum setuju, kembalikan ke halaman home
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

        # 6. Jika klasifikasi adalah IGA0, lakukan deteksi wajah
        if predicted_label == "iga0":  # Jika IGA0, cek ada wajah atau tidak
            print("[DEBUG] Prediksi adalah IGA0, memeriksa wajah...")
            if not detect_face(filepath):  # Jika tidak ada wajah terdeteksi
                print("[DEBUG] Tidak ada wajah yang terdeteksi.")
                # Kirimkan status error ke template untuk memunculkan pop-up
                return render_template('classify.html', error_message="Tidak ada wajah terdeteksi. Pastikan gambar mengandung wajah untuk analisis jerawat.")
        
        # 7. Redirect ke halaman hasil
        return redirect(url_for('result', image=filename, label=predicted_label))

    # Kalau GET, tampilkan halaman form upload
    return render_template('classify.html')

@app.route('/terms-accepted')
def terms_accepted():
    session['terms_accepted'] = True  # Tandai bahwa pengguna telah setuju
    return redirect(url_for('classify'))  # Redirect ke halaman classify

@app.route('/result')
def result():
    image = request.args.get('image')
    label = request.args.get('label')
    return render_template('result.html', image=image, label=label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
