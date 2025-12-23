import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout,
    Dense, Flatten, Activation,
)
from scikeras.wrappers import KerasClassifier

from src.data.config import IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES


def build_cnn(
    epochs=1,
    batch_size=10,
    layers=5,
    dropout=0.5,
    activation="relu"
):
    def create_model():
        model = Sequential()

        model.add(
            Conv2D(
                64,
                (3, 3),
                padding="same",
                activation=activation,
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
            )
        )

        for _ in range(layers - 1):
            model.add(Conv2D(64, (3, 3), padding="same"))
            model.add(Activation(activation))

        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout / 2))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation(activation))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout / 2))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(activation))
        model.add(Dropout(dropout))

        model.add(Dense(NUM_CLASSES, activation="softmax"))

        opt = keras.optimizers.RMSprop(learning_rate=1e-4)

        model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(
                    multi_label=True,
                    num_labels=NUM_CLASSES,
                    name="auc",
                ),
            ],
        )

        return model

    return KerasClassifier(
        model=create_model,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )