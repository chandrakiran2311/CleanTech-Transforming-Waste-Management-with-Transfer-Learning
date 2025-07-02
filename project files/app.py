from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Load the trained model
model = load_model("vgg16_model.keras")

# Your label mapping (example)
class_names = ['biodegradable', 'recyclable', 'trash']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print(">> POST request received")

        if 'file' not in request.files:
            print(">> No file part")
            return "No file part"

        file = request.files['file']
        if file.filename == '':
            print(">> No file selected")
            return "No file selected"

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            print(f">> Saving file to: {filepath}")
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # normalize

            prediction = model.predict(img_array)
            predicted_label = class_names[np.argmax(prediction)]
            print(f">> Predicted label: {predicted_label}")

            return render_template("index.html", prediction=predicted_label, image_path=filepath)

    return render_template("index.html")
if __name__ == "__main__":
    print(">> Flask app is starting...")
    app.run(debug=True)
    print(">> Flask app has stopped.")
