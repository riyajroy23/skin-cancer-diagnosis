import tensorflow as tf
from src.models.transfer_learning import transfer_learning_model
from src.data.dataset import load_dataset
from src.data.preprocessing import one_hot


def train_transfer_learning():
    X_train, X_val, _, _, y_train, y_val = load_dataset(
        image_dir="raw-data/images",
        metadata_csv="src/data/HAM10000_metadata.csv",
        test_size=0.2,
    )

    model = transfer_learning_model(train_base=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/transfer_vgg16_best.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
        X_train,
        one_hot(y_train),
        validation_data=(X_val, one_hot(y_val)),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    train_transfer_learning()