import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout,
    Dense, Flatten, Activation,
    Reshape, BatchNormalization
)
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from src.dataset.config import IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES

def CNNClassifier(epochs=1, batch_size=10, layers=5, dropout=0.5, activation="relu"):

    def create_model():
        model = Sequential()
        model.add(Reshape((IMG_WIDTH, IMG_HEIGHT, 3)))

        for _ in range(layers):
            model.add(Conv2D(64, (3, 3), padding="same"))
            model.add(Activation(activation))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout / 2))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation(activation))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout / 2))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
        model.add(Dense(NUM_CLASSES, activation="softmax"))

        opt = keras.optimizers.RMSprop(learning_rate=1e-4, decay=1e-6)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=[tf.keras.metrics.AUC()]
        )

        return model

    return KerasClassifier(
        build_fn=create_model,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

def transfer_learning_model():
    base = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling="max"
    )

    model = Sequential([
        base,
        Dropout(0.1),
        BatchNormalization(),
        Dense(256, activation="relu"),
        Dropout(0.1),
        BatchNormalization(),
        Dense(NUM_CLASSES, activation="softmax")
    ])

    opt = keras.optimizers.RMSprop(learning_rate=1e-4, decay=1e-6)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=[tf.keras.metrics.AUC()]
    )

    return model
