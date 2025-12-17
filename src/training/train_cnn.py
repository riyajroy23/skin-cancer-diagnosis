import numpy as np
from models import CNNClassifier
from data import one_hot, augment_data, load_dataset

X_train, X_test, Xg_train, Xg_test, y_train, y_test = load_dataset(
    image_dir="data/images",
    metadata_csv="data/HAM10000_metadata.csv"
)

X_train, y_train = augment_data(X_train, y_train)

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

cnn = CNNClassifier()
cnn.fit(
    X_train.astype(np.float32),
    y_train_oh.astype(np.float32),
    validation_data=(X_test.astype(np.float32), y_test_oh.astype(np.float32))
)
