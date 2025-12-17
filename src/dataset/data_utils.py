import os
import cv2
import numpy as np

IMG_WIDTH = 100
IMG_HEIGHT = 75

def load_images(image_paths, labels):
    X = []
    X_gray = []
    y = []

    for img_path, label in zip(image_paths, labels):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        X.append(img)
        X_gray.append(gray)
        y.append(label)

    return np.array(X), np.array(X_gray), np.array(y)