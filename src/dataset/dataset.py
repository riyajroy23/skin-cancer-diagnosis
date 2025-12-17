from sklearn.model_selection import train_test_split
from data_loader import load_ham10000
from augmentation import augment_dataset
from .config import RANDOM_STATE

def load_dataset(image_dir, metadata_csv, augment=True):
    X, X_gray, y = load_ham10000(image_dir, metadata_csv)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )

    Xg_train, Xg_test, _, _ = train_test_split(
        X_gray, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )

    if augment:
        X_train, Xg_train, y_train = augment_dataset(
            X_train, Xg_train, y_train, iterations=1
        )

    return X_train, X_test, Xg_train, Xg_test, y_train, y_test