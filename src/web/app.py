import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

from src.data.config import IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES

CLASS_EXPLANATIONS = {
    "mel": {
        "name": "Melanoma",
         "description": (
            "Melanoma is a serious form of skin cancer that develops in melanocytes. "
            "It can grow rapidly and spread to other parts of the body if not detected early. "
            "Common warning signs include asymmetry, irregular borders, color variation, "
            "large diameter, or changes over time."
        )
    },

    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": (
            "Basal cell carcinoma is the most common type of skin cancer. "
            "It usually grows slowly and rarely spreads, but early detection is important."
        )
    },

    "bkl": {
        "name": "Benign Keratosis",
        "description": (
            "These are non-cancerous skin growths such as seborrheic keratoses. "
            "They are generally harmless but can resemble malignant lesions."
        )
    },

    "df": {
        "name": "Dermatofibroma",
        "description": (
            "Dermatofibromas are benign skin nodules often caused by minor injuries like insect bites. "
            "They are typically firm and slow-growing."
        )
    },

    "mel": {
        "name": "Melanoma",
        "description": (
            "Melanoma is a serious form of skin cancer that can spread rapidly if not treated early. "
            "Early detection significantly improves outcomes."
        )
    },

    "nv": {
        "name": "Melanocytic Nevus",
        "description": (
            "Melanocytic nevi are common moles that are usually benign. "
            "However, changes in size, color, or shape should be evaluated by a professional."
        )
    },

    "vasc": {
        "name": "Vascular Lesions",
        "description": (
            "Vascular lesions are related to blood vessels, such as angiomas or hemorrhages. "
            "They are usually benign and not cancerous."
        )
    }
}

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

model = tf.keras.models.load_model(
    "models/transfer_vgg16_best.keras"
)


def preprocess_image(image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file).convert("RGB")

    processed = preprocess_image(image)
    probs = model.predict(processed)[0]

    class_idx = int(np.argmax(probs))
    label = CLASS_NAMES[class_idx]
    confidence = float(probs[class_idx])

    explanation = CLASS_EXPLANATIONS[label]
    print(explanation)

    return jsonify({
        "label": label,
        "name": explanation["name"],
        "description": explanation["description"],
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)