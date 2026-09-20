import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
)


class ModelEvaluator:
    # Hitung metrik evaluasi + laporan (classification report & confusion matrix).

    def __init__(self, class_names):
        self.class_names = list(class_names)

    def metrics(self, y_true, y_pred):
        # Dict metrik utama. Macro F1 = metrik utama karena target imbalanced.
        return {
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_true, y_pred),
        }

    def text_report(self, y_true, y_pred):
        # Classification report + confusion matrix dalam bentuk teks.
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        return f"Classification Report\n{report}\n\nConfusion Matrix\n{cm_df}"
