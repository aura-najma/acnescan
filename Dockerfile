# Gunakan image Python resmi yang ringan
FROM python:3.10-slim

# Install library sistem yang dibutuhkan oleh OpenCV dan Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja di container
WORKDIR /app

# Salin seluruh isi folder app ke container
COPY app/ /app/

# (Opsional redundancy, ini bisa dihapus kalau sudah di atas)
# COPY app/model /app/model

# Buat folder static/uploads agar tidak error saat simpan file
RUN mkdir -p /app/static/uploads

# Install dependency
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port Flask
EXPOSE 5000

# Jalankan aplikasi
CMD ["python", "app.py"]
