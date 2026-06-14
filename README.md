# AcneScan

AcneScan is a web-based acne detection application developed as a final project for the **Machine Learning Course (Semester 6, Academic Year 2024/2025)**. The application utilizes the **YOLO (You Only Look Once)** object detection model to identify acne lesions from facial images and presents the results through an interactive web interface built with Flask.

## Authors

This project was developed by:

* Aura Najma Kustiananda
* Elzandi Irfan Zikra
* Putu Angga Kurniawan
* Brilliant Edgar Prasetyo
* Ditha Meiga Zakaria

## Academic Information

**Course:** Machine Learning
**Semester:** 6
**Academic Year:** 2024/2025

## Project Background

Acne is one of the most common skin conditions affecting people of various age groups. Advances in artificial intelligence and computer vision have enabled the development of automated systems capable of detecting visual skin conditions from digital images.

AcneScan was created to demonstrate the implementation of machine learning concepts in a real-world healthcare-related use case. By leveraging the YOLO object detection model, the system can analyze facial images and identify acne regions automatically.

## Features

* Acne detection from facial images
* YOLO-based object detection model
* Web application built using Flask
* Image upload and prediction functionality
* Visualized detection results

## Technology Stack

* Python
* Flask
* YOLO
* OpenCV
* NumPy
* HTML/CSS
* JavaScript

## System Workflow

1. User uploads a facial image through the web interface.
2. The image is processed by the Flask backend.
3. The YOLO model performs acne detection.
4. Detected acne areas are marked on the image.
5. The prediction result is displayed to the user.

## Installation

### Clone Repository

```bash
git clone https://github.com/aura-najma/acnescan.git
cd acnescan
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open your browser and access:

```text
http://localhost:5000
```

## Project Objectives

* Apply machine learning concepts learned during the course.
* Explore computer vision techniques for image analysis.
* Implement YOLO for object detection tasks.
* Develop a functional web application integrating AI models.
* Demonstrate practical applications of machine learning in healthcare-related scenarios.

## Future Improvements

* Acne severity classification
* Support for multiple skin conditions
* Mobile application integration
* Improved model accuracy with larger datasets
* Cloud deployment and API integration

## Disclaimer

This project was developed for academic and educational purposes as part of the Machine Learning course requirements. The application is not intended to provide medical diagnoses and should not replace consultation with healthcare professionals.

## License

This repository is intended for educational use. Please contact the authors for any questions regarding reuse or distribution.
