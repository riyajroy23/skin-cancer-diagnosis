import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .augmentation import augment_data
from .config import RANDOM_STATE, IMG_WIDTH, IMG_HEIGHT

# HAM10000 class mapping
CLASS_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}

def load_ham10000(image_dir, metadata_csv):
    """
    Loads raw HAM10000 images and labels from disk.
    Returns:
        X       : RGB images (N, H, W, 3)
        X_gray  : Grayscale images (N, H, W)
        y       : Integer labels (N,)
    """
    df = pd.read_csv(metadata_csv)

    X, X_gray, y = [], [], []

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, f"{row['image_id']}.jpg")
        label = CLASS_MAP[row["dx"]]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        X.append(img)
        X_gray.append(gray)
        y.append(label)

    return np.array(X), np.array(X_gray), np.array(y)

def load_dataset(image_dir, metadata_csv, test_size=0.4, augment=True):
    """
    End-to-end dataset preparation pipeline:
    - Load HAM10000 images
    - Train/test split
    - Optional data augmentation
    """
    X, X_gray, y = load_ham10000(image_dir, metadata_csv)

    X_train, X_test, Xg_train, Xg_test, y_train, y_test = train_test_split(
        X, X_gray, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    if augment:
        X_train, Xg_train, y_train = augment_data(
            X_train, Xg_train, y_train, iterations=1
        )

    return X_train, X_test, Xg_train, Xg_test, y_train, y_test
