import numpy as np

from src.data.dataset import load_dataset
from src.data.preprocessing import one_hot
from src.data.config import CLASS_NAMES
from src.eval.metrics import model_stats
from src.models.cnn import build_cnn

def main():
    X_train, X_test, _, _, y_train, y_test = load_dataset(
        image_dir="raw-data/images",
        metadata_csv="src/data/HAM10000_metadata.csv"
    )

    # One-hot encode labels
    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)

    # Build and train CNN
    cnn = build_cnn()
    cnn.fit(
        X_train.astype(np.float32),
        y_train_oh.astype(np.float32),
        validation_data=(
            X_test.astype(np.float32),
            y_test_oh.astype(np.float32)
        ),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # ---- Evaluation ----
    y_pred_probalities = cnn.predict(X_test.astype(np.float32))
    y_pred = y_pred_probalities.argmax(axis=1)

    model_stats(
        name="CNN Classifier",
        y_true=y_test,
        y_pred=y_pred,
        y_pred_probabilities=y_pred_probalities,
    )

if __name__ == "__main__":
    main()
