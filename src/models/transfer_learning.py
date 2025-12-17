import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from models import transfer_learning_model
from data import one_hot

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

transfer = KerasClassifier(
    build_fn=transfer_learning_model,
    epochs=1,
    verbose=1
)

transfer.fit(
    X_train.astype(np.float32),
    y_train_oh.astype(np.float32),
    validation_data=(X_test.astype(np.float32), y_test_oh.astype(np.float32))
)