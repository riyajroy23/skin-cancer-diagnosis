import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from .config import IMG_WIDTH, IMG_HEIGHT, RANDOM_STATE

# class mapping used in HAM10000
CLASS_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}

def load_dataset(image_dir, metadata_csv):
    import pandas as pd

    df = pd.read_csv(metadata_csv)

    X = []
    X_gray = []
    y = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row["image_id"] + ".jpg")
        label = CLASS_MAP[row["dx"]]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        X.append(img)
        X_gray.append(gray)
        y.append(label)

    return np.array(X), np.array(X_gray), np.array(y)