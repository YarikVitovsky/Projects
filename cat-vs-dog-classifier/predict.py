import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys

# Load the trained model
model = load_model("cat_dog_model.h5")

# Image path — change this or pass via command line
img_path = "Test_mia.jpg"

# Load and preprocess the image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
confidence = float(prediction[0][0])
label = "Dog" if confidence > 0.5 else "Cat"
confidence_pct = confidence if confidence > 0.5 else 1 - confidence

print(f"Prediction: {label} ({confidence_pct * 100:.2f}% confidence)")
