## 🐶🐱 Cat vs Dog Image Classifier

A simple image classification app that predicts whether an image contains a cat or a dog — using transfer learning with VGG16, and deployed on Hugging Face Spaces using Gradio.

---

### 🔍 What It Does
- VGG16 model pretrained on ImageNet, with custom classifier layers on top
- Classifies images as **Cat** or **Dog** with 93.10% confidence
- Publicly deployed as a live web app
- Drag-and-drop interface + example image

---

### 🚀 Live Demo
👉 Try the classifier on Hugging Face: https://huggingface.co/spaces/yarikvitovsky/cat-vs-dog-classifier

---

### 🧠 What I Learned
I followed a guided, tutorial-style implementation to build this project from end to end.
Along the way, I:

- Got hands-on experience with VGG16, one of the most well-known pre-trained models in computer vision
- Learned to use TensorFlow and Keras to build and train models
- Trained a model to solve a real task: classifying images of cats and dogs
- Worked with a dataset of 2,000+ labeled images
- Built a complete ML pipeline — from data preprocessing to training to deploying a live web app
- Deployed the trained model with Gradio on Hugging Face Spaces, allowing real-time predictions through a simple UI

---

### 📦 Files
- `app.py` — Gradio web app
- `cat_dog_model.h5` — saved model
- `test_mia.jpg` — example image shown by default
- `requirements.txt` — for Hugging Face deployment

---

### 🛠 Run Locally

```bash
pip install gradio tensorflow pillow numpy
python app.py
