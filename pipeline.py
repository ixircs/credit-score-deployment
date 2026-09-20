import argparse
import os
import tempfile
from datetime import datetime

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn

from data_ingestion import ingest_data
from pre_processing import DataPreprocessor
from train import ModelTrainer
from evaluation import ModelEvaluator

ARTIFACT_DIR = "artifacts"
# Simpan eksperimen MLflow di SQLite (bukan folder ./mlruns yang sudah deprecated).
# Ini juga default `mlflow ui`, jadi cukup jalankan `mlflow ui` untuk melihatnya.
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def log_text_artifact(text, filename):
    # Tulis teks ke file sementara lalu log sebagai artifact MLflow.
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    mlflow.log_artifact(path)
    os.remove(path)


def run_pipeline(data_path, output_path, n_iter, test_size, experiment_name):
    print("Step 1: Data Ingestion")
    df_raw = ingest_data(data_path)

    print("Step 2: Preprocessing")
    prep = DataPreprocessor()
    df_clean = prep.clean(df_raw)
    X, y_raw = prep.split_features_target(df_clean)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()
    print(f"  Fitur: {X.shape[1]} kolom | Kelas: {class_names}")

    transformer = prep.build_transformer(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    trainer = ModelTrainer(transformer)
    evaluator = ModelEvaluator(class_names)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    print("Step 3: Training baseline (beberapa model)")
    baseline_scores = {}
    for name in trainer.registry:
        with mlflow.start_run(run_name=f"baseline_{name}"):
            pipe = trainer.baseline_fit(name, X_tr, y_tr)
            m = evaluator.metrics(y_te, pipe.predict(X_te))
            baseline_scores[name] = m["macro_f1"]
            mlflow.set_tag("stage", "baseline")
            mlflow.log_param("model", name)
            mlflow.log_metrics(m)
            print(f"  {name:20s} Macro F1 = {m['macro_f1']:.4f}")

    # Ambil top 3 model terbaik untuk di-tuning.
    top3 = sorted(baseline_scores, key=baseline_scores.get, reverse=True)[:3]
    print(f"  Top 3 untuk tuning: {top3}")

    print(f"Step 4: Tuning (RandomizedSearchCV, n_iter={n_iter})")
    best = {"name": None, "macro_f1": -1.0, "pipeline": None, "params": None}
    for name in top3:
        with mlflow.start_run(run_name=f"tuned_{name}"):
            search = trainer.tune(name, X_tr, y_tr, n_iter=n_iter)
            tuned_pipe = search.best_estimator_
            y_pred = tuned_pipe.predict(X_te)
            m = evaluator.metrics(y_te, y_pred)

            mlflow.set_tag("stage", "tuned")
            mlflow.log_param("model", name)
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_macro_f1", search.best_score_)
            mlflow.log_metrics(m)
            log_text_artifact(evaluator.text_report(y_te, y_pred),
                              f"report_{name.replace(' ', '_')}.txt")

            print(f"  {name:20s} test Macro F1 = {m['macro_f1']:.4f} "
                  f"(CV {search.best_score_:.4f})")
            if m["macro_f1"] > best["macro_f1"]:
                best.update(name=name, macro_f1=m["macro_f1"],
                            pipeline=tuned_pipe, params=search.best_params_)

    print(f"Step 5: Simpan model terbaik -> {best['name']} "
          f"(Macro F1 = {best['macro_f1']:.4f})")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    bundle = {
        "pipeline": best["pipeline"],
        "label_encoder": le,
        "num_cols": prep.num_cols,
        "cat_cols": prep.cat_cols,
        "feature_order": X.columns.tolist(),
        "classes": class_names,
        "model_name": best["name"],
        "best_params": best["params"],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, output_path, compress=3)
    size_mb = round(os.path.getsize(output_path) / 1e6, 1)
    print(f"  Artefak tersimpan -> {output_path} ({size_mb} MB)")

    # Catat model terbaik sebagai satu run final di MLflow.
    with mlflow.start_run(run_name="BEST_MODEL"):
        mlflow.set_tag("stage", "best")
        mlflow.log_param("model", best["name"])
        mlflow.log_params(best["params"])
        mlflow.log_metric("macro_f1", best["macro_f1"])
        mlflow.sklearn.log_model(best["pipeline"], name="model")

    return best


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline training Credit Score")
    p.add_argument("--data", default="data_A.csv")
    p.add_argument("--output", default=os.path.join(ARTIFACT_DIR, "model_credit_score.pkl"))
    p.add_argument("--n-iter", type=int, default=8)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--experiment", default="credit_score_dataset_A")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.data, args.output, args.n_iter, args.test_size, args.experiment)
