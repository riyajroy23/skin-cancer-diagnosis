import numpy as np

def flatten_images(X):
    """
    Flattens image tensors for classical ML models (e.g. KNN).
    Input:  (N, H, W, C)
    Output: (N, H*W*C)
    """
    return X.reshape(X.shape[0], -1)


def one_hot(y, num_classes=None):
    """
    Convert integer labels to one-hot encoded vectors.

    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Integer class labels.
    num_classes : int, optional
        Total number of classes. If None, inferred from y.

    Returns
    -------
    np.ndarray, shape (n_samples, num_classes)
        One-hot encoded labels.
    """
    y = np.asarray(y).astype(int)

    if num_classes is None:
        num_classes = np.max(y) + 1

    return np.eye(num_classes)[y]