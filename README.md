🌿 Soyabean Leaf Disease Detection
📖 Overview

This project focuses on detecting and classifying diseases in soyabean leaves using deep learning techniques. The system analyzes uploaded leaf images and predicts whether they are healthy or affected by a specific disease, helping farmers and researchers identify plant issues early and take preventive measures.

⚙️ Features

Detects and classifies multiple soyabean leaf diseases

User-friendly web interface for image upload and prediction

Uses a trained deep learning model (CNN/YOLOv8) for accurate results

Fast and lightweight for deployment on cloud platforms

🧠 Technologies Used

Python 3.9

Flask – for web framework

TensorFlow / Keras / PyTorch – for deep learning model

OpenCV, NumPy, Pandas, Matplotlib – for image processing and analysis

HTML, CSS (Flask templates) – for frontend interface

🧩 Project Structure
soyabean_leaf_disease_detection_main/
│
├── app_main.py              # Main Flask application
├── requirements.txt         # Dependencies
├── models/                  # Trained model files (.h5 / .pkl)
├── static/                  # CSS, JS, and images for UI
├── templates/               # HTML templates
└── README.md                # Project documentation

🚀 How to Run Locally

Clone the repository

git clone https://github.com/<your-username>/soyabean-leaf-disease-detection.git
cd soyabean-leaf-disease-detection


Install dependencies

pip install -r requirements.txt


Run the app

python app_main.py


Open in browser

http://127.0.0.1:5000/

📸 Usage

Upload a soyabean leaf image through the web interface.

The system processes the image using the trained model.

It displays the predicted disease type or shows “Healthy Leaf.”

🧪 Model Training (Optional)

If you want to retrain or fine-tune the model, use the provided Jupyter notebook:

soyaleaf_detection_main.ipynb

📁 Notes

dataset/ and uploads/ folders are excluded from the repo to reduce size.

You can add your dataset for retraining if needed.

Works best with Python 3.8+ and TensorFlow/Keras installed.

👨‍💻 Author

Akhilesh Chitare
📍 Nagpur, Maharashtra, India
📧 akhileshchitare04@gmail.com

🔗 linkedin.com/in/akhilesh00
