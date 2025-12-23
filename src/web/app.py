import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

from src.data.config import IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES

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
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file).convert("RGB")

        processed = preprocess_image(image)
        probs = model.predict(processed)[0]

        class_idx = np.argmax(probs)
        prediction = CLASS_NAMES[class_idx]
        confidence = float(probs[class_idx])

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
    )


if __name__ == "__main__":
    app.run(debug=True)
