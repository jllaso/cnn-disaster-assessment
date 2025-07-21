# cnn-disaster-assessment
# Damage Assessment Using Convolutional Neural Networks

This project is based on my Master's thesis, where I developed and evaluated CNN-based models to assess the impact of natural disasters on infrastructures using satellite imagery.

### 🧠 Model Objective

Given two satellite images (pre- and post-disaster), the model classifies the damage level on buildings using Siamese CNN architectures with transfer learning.

### 📊 Key Results

- **94.2% accuracy** (binary classification) using InceptionResNetV2
- Models compared: VGG-16, InceptionV3, ResNet50
- Dataset: [xView2](https://xview2.org/)
- Libraries: TensorFlow, Keras, OpenCV, NumPy

### 📁 Structure

- `notebook/`: contains the Jupyter Notebook with data preprocessing, model training and evaluation
- `assets/`: example satellite image pairs
- `requirements.txt`: list of required Python packages

### 📌 What You’ll Learn

- How to build and evaluate CNNs for image classification
- How to use pre-trained architectures in transfer learning
- Insights on handling real-world satellite imagery data

### 🧑‍🎓 Background

This work was part of my MSc in Electrical and Computer Engineering at the Illinois Institute of Technology (Chicago).
