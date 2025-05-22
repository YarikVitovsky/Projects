import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model("cat_dog_model.h5")


# Prediction function
def predict_img(img):
    img = img.resize((150, 150))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)[0][0]
    return "Dog 🐶" if pred > 0.5 else "Cat 🐱"


# Gradio interface
gr.Interface(
    fn=predict_img,
    inputs="image",
    outputs="text",
    title="Cat vs Dog Classifier",
    description="Upload an image to classify it as a cat or a dog."
).launch()
