from src.dataset.data_loader import load_dataset
from src.models.models import flatten_images, build_knn
from evaluation import model_stats

X_train, X_test, Xg_train, Xg_test, y_train, y_test = load_dataset(
    image_dir="dataset/images",
    metadata_csv="dataset/HAM10000_metadata.csv"
)

Xg_train_flat = flatten_images(Xg_train)
Xg_test_flat = flatten_images(Xg_test)

knn = build_knn(n_neighbors=5)
knn.fit(Xg_train_flat, y_train)

y_pred = knn.predict(Xg_test_flat)
y_pred_proba = knn.predict_proba(Xg_test_flat)

model_stats("KNN Classifier", y_test, y_pred, y_pred_proba)
