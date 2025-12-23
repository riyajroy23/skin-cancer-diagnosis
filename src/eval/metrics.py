import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from src.data.config import CLASS_NAMES

def model_stats(name, y_true, y_pred, y_pred_probabilities):
    print(name)

    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(
        y_true,
        y_pred_probabilities,
        multi_class="ovo",
        average="macro"
    )

    print(f"Accuracy: {acc:.5f}")
    print(f"ROC AUC:  {roc:.5f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.show()

    return cm
