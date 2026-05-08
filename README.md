# Automated Visual Classification using Deep Learning

## Project Overview

This project implements an AI-powered image classification system using Deep Learning and Computer Vision techniques. The model classifies natural scenes into multiple categories such as buildings, forest, glacier, mountain, sea, and street using Transfer Learning with MobileNetV2.

The project demonstrates:
- Image preprocessing
- Data augmentation
- Transfer learning
- Model training
- Real-time prediction
- Deep learning deployment workflow

This project is designed as a production-style Computer Vision pipeline suitable for academic projects, portfolios, and machine learning practice.

---

# Dataset

Dataset Used:
Intel Image Classification Dataset

Classes:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

Dataset Link:
https://www.kaggle.com/datasets/puneet6060/intel-image-classification

---

# Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Google Colab
- MobileNetV2

---

# Project Structure

```bash
Automated-Visual-Classification/
│
├── dataset/
│
├── notebook/
│   └── Automated_Visual_Classification.ipynb
│
├── models/
│   └── scene_classification_model.h5
│
├── app/
│   └── app.py
│
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Workflow

## 1. Data Preprocessing
- Resize images
- Normalize pixel values
- Apply data augmentation

## 2. Model Building
- Use MobileNetV2 pre-trained model
- Freeze base layers
- Add custom dense layers

## 3. Model Training
- Train using augmented dataset
- Validate model performance

## 4. Prediction
- Predict scene category from uploaded image

## 5. Deployment
- Flask API for inference
- Real-time image classification support

---

# Model Architecture

The project uses:
- MobileNetV2 (Transfer Learning)
- GlobalAveragePooling2D
- Dense Layers
- Dropout Layer
- Softmax Activation

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Automated-Visual-Classification.git
cd Automated-Visual-Classification
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Run:

```bash
python train.py
```

Model file will be saved inside:

```bash
models/
```

---

# Prediction

Place test image in project folder and run:

```bash
python predict.py
```

---

# Flask API

Run:

```bash
python app/app.py
```

API Endpoint:

```bash
POST /predict
```

---

# Results

Expected Accuracy:
- 85% to 95%

The model performs multi-class scene classification using transfer learning and achieves strong validation accuracy.

---

# Future Improvements

- Grad-CAM visualization
- Real-time webcam prediction
- Docker deployment
- Streamlit dashboard
- Model optimization
- Edge AI deployment

---

# Learning Outcomes

This project demonstrates:
- Deep Learning fundamentals
- CNN architecture
- Transfer Learning
- Image preprocessing
- Computer Vision workflows
- Model deployment basics

---

# Applications

- Smart surveillance
- Autonomous systems
- Environmental monitoring
- Scene understanding
- Industrial visual AI systems

---

# Author

Sampath

Machine Learning & Computer Vision Project
