import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import model_from_json

# Load model architecture
with open('model_a.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load weights
model.load_weights('model_weights.weights.h5')

# Define emotion mapping
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Global variables to hold the uploaded image and its path
uploaded_image = None
uploaded_image_path = None

def upload_image():
    global uploaded_image, uploaded_image_path

    filepath = filedialog.askopenfilename()
    if not filepath:
        return

    uploaded_image_path = filepath

    # Load and display image in GUI
    img = Image.open(filepath)
    img.thumbnail((200, 200))
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img

    # Clear emotion label
    emotion_label.config(text="")

def detect_emotion():
    global uploaded_image_path

    if not uploaded_image_path:
        messagebox.showerror("Error", "Please upload an image first.")
        return

    # Load and preprocess image
    image = cv2.imread(uploaded_image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = image / 255.0

    # Predict emotion
    prediction = model.predict(image)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]

    # Display prediction result
    emotion_label.config(text=emotion)

# Create Tkinter window
window = tk.Tk()
window.title("Emotion Detector")
window.geometry("400x400")

# Add a label
title_label = tk.Label(window, text="Emotion Detector", font=("Arial", 24))
title_label.pack(pady=20)

# Add a label to display emotion
emotion_label = tk.Label(window, text="", font=("Arial", 18))
emotion_label.pack(pady=10)

# Add a panel to display the uploaded image
panel = tk.Label(window)
panel.pack(pady=10)

# Add a button to upload image
upload_button = tk.Button(window, text="Upload an Image", command=upload_image, font=("Arial", 16))
upload_button.pack(pady=10)

# Add a button to detect emotion
detect_button = tk.Button(window, text="Detect Emotion", command=detect_emotion, font=("Arial", 16))
detect_button.pack(pady=10)

# Run the application
window.mainloop()
