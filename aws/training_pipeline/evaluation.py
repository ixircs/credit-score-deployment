# Tahap 4 SageMaker Pipeline: evaluation.
# Baca model (model.tar.gz dari TrainingStep) + test.csv, hitung Macro F1 & accuracy,
# tulis evaluation.json (dibaca PropertyFile/JsonGet di pipeline).
import os
import json
import tarfile
import joblib
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

TARGET = "Credit_Score"

if __name__ == "__main__":
    if os.path.exists("/opt/ml/processing"):
        model_dir = "/opt/ml/processing/model"
        test_path = "/opt/ml/processing/test/test.csv"
        out_dir = "/opt/ml/processing/evaluation"
    else:
        base = "/home/ec2-user/SageMaker/credit_pipeline"
        model_dir = f"{base}/model"
        test_path = f"{base}/test/test.csv"
        out_dir = f"{base}/eval"

    os.makedirs(out_dir, exist_ok=True)

    # TrainingStep menaruh model sebagai model.tar.gz -> ekstrak dulu.
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path) as tar:
            tar.extractall(model_dir)
    bundle = joblib.load(os.path.join(model_dir, "model_credit_score.joblib"))

    df = pd.read_csv(test_path)
    X = df.drop(columns=[TARGET])
    y_true = bundle["label_encoder"].transform(df[TARGET])
    y_pred = bundle["pipeline"].predict(X)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    report = {
        "multiclass_classification_metrics": {
            "macro_f1": {"value": float(macro_f1), "standard_deviation": "NaN"},
            "accuracy": {"value": float(acc), "standard_deviation": "NaN"},
        }
    }
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(report, f)

    print(f"Evaluation OK: Macro F1 = {macro_f1:.4f} | Accuracy = {acc:.4f}")
