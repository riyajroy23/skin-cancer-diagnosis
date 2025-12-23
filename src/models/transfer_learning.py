import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, BatchNormalization
)

from src.data.config import IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES


def transfer_learning_model(train_base=False):
    base = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    )

    base.trainable = train_base

    model = Sequential([
        base,
        Flatten(),
        BatchNormalization(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        BatchNormalization(),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

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
