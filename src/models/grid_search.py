from hypopt import GridSearch
from models import CNNClassifier
from data import one_hot

param_grid = {
    "epochs": [10, 20],
    "batch_size": [32, 64],
    "layers": [1, 3],
    "dropout": [0.3, 0.5],
    "activation": ["relu"]
}

def gridSearchCNN():
    return CNNClassifier()

gs = GridSearch(
    model=gridSearchCNN(),
    param_grid=param_grid,
    parallelize=False
)

gs.fit(X_train, one_hot(y_train), X_val, one_hot(y_val))