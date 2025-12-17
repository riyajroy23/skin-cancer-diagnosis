import cv2
import random
import numpy as np

def flip_image(img, axis=1):
    return cv2.flip(img, axis)

def blur_image(img, kx=8, ky=10):
    return cv2.blur(img, (kx, ky))

def zoom_image(img, zoom=0.33):
    h, w = img.shape[:2]
    centerX, centerY = h // 2, w // 2

    radiusX = int((1 - zoom) * h)
    radiusY = int((1 - zoom) * w)

    minX, maxX = centerX - radiusX, centerX + radiusX
    minY, maxY = centerY - radiusY, centerY + radiusY

    cropped = img[minX:maxX, minY:maxY]
    return cv2.resize(cropped, (w, h))


def augment_dataset(X, X_gray, y, iterations=1):
    X_aug, Xg_aug, y_aug = [], [], []

    for _ in range(iterations):
        for i in range(len(X)):
            choice = random.randint(0, 2)

            if choice == 0:
                X_aug.append(flip_image(X[i], 1))
                Xg_aug.append(flip_image(X_gray[i], 1))
            elif choice == 1:
                X_aug.append(blur_image(X[i]))
                Xg_aug.append(blur_image(X_gray[i]))
            else:
                X_aug.append(zoom_image(X[i]))
                Xg_aug.append(zoom_image(X_gray[i]))

            y_aug.append(y[i])

    return (
        np.vstack((X, np.array(X_aug))),
        np.vstack((X_gray, np.array(Xg_aug))),
        np.append(y, np.array(y_aug))
    )
