🐶🐱 Cat vs Dog Image Classifier
A simple image classification app that predicts whether an image contains a cat or a dog — using transfer learning with VGG16, and deployed on Hugging Face Spaces using Gradio.

🔍 What It Does
Fine-tuned VGG16 model (pretrained on ImageNet)

Classifies uploaded images as Cat or Dog

Returns prediction with 93.10% confidence

Publicly deployed as a live web app

Drag-and-drop interface + example image (🐱 This is my cat Mia!)

🚀 Live Demo
👉 Try the classifier on Hugging Face <!-- Replace with actual link -->

🧠 What I Learned
I followed a tutorial-style implementation, but made sure I understood how data preprocessing, model loading, and deployment works. The app loads a real trained model and makes live predictions through a web interface — and I fixed a real deployment error along the way!

📦 Files
app.py — Gradio web app

cat_dog_model.h5 — trained VGG16 model

test_mia.jpg — example image used by default

requirements.txt — for Hugging Face deployment


## 🛠 Run Locally

```bash
pip install gradio tensorflow pillow numpy
python app.py
