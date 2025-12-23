from src.data.dataset import load_dataset
from src.data.preprocessing import flatten_images
from src.models.knn import build_knn
from src.eval.metrics import model_stats

X_train, X_test, Xg_train, Xg_test, y_train, y_test = load_dataset(
    image_dir="raw-data/images",
    metadata_csv="src/data/HAM10000_metadata.csv"
)

Xg_train_flat = flatten_images(Xg_train)
Xg_test_flat = flatten_images(Xg_test)

knn = build_knn(n_neighbors=5)
knn.fit(Xg_train_flat, y_train)

y_pred = knn.predict(Xg_test_flat)
y_pred_probalities = knn.predict_proba(Xg_test_flat)

model_stats("KNN Classifier", y_test, y_pred, y_pred_probalities)
