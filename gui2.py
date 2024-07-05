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

# Define function to upload image
def upload_image():
    filepath = filedialog.askopenfilename()
    if not filepath:
        return

    # Load and preprocess image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = image / 255.0

    # Predict
    prediction = model.predict(image)
    emotion_index = np.argmax(prediction)
    emotion_label = emotion_labels[emotion_index]

    # Display prediction result
    messagebox.showinfo("Prediction", "Emotion: " + emotion_label)

    # Display image in GUI
    img = Image.open(filepath)
    img.thumbnail((200, 200))
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img

# Create Tkinter window
window = tk.Tk()
window.title("Emotion Detector")
window.geometry("400x400")

# Add a label
label = tk.Label(window, text="Emotion Detector", font=("Arial", 24))
label.pack(pady=20)

# Add a button to upload image
button = tk.Button(window, text="Upload Image", command=upload_image, font=("Arial", 16))
button.pack(pady=20)

# Add a panel to display the uploaded image
panel = tk.Label(window)
panel.pack(pady=20)

# Run the application
window.mainloop()
