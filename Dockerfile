# Gunakan image Python resmi yang ringan
FROM python:3.10-slim

# Install semua library sistem yang dibutuhkan OpenCV + Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Salin semua isi folder /app lokal ke container
COPY app/ /app/
COPY app/model /app/model

# Install semua package dari requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Buka port Flask
EXPOSE 5000

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
