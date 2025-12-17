from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

def model_stats(name, y_true, y_pred, y_pred_proba):
    print(name)

    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_proba, multi_class="ovo")

    print(f"Accuracy: {acc:.5f}")
    print(f"ROC AUC:  {roc:.5f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.show()

    return confusion_matrix(y_true, y_pred)
