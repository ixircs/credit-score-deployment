# generate_artifacts.py (OPSIONAL) — isi folder ingested/train/test/model/eval.
# =============================================================================
# Menjalankan logika tahap yang SAMA seperti pipeline 2a (ingest -> preprocess ->
# train -> evaluate) tapi berurutan dalam 1 proses, sehingga artefak tiap tahap
# BENAR-BENAR tersimpan ke folder (bukan di folder sementara SageMaker local mode).
# Berguna untuk "penampakan" artefak di laporan/video. Tidak mengubah bukti 2a
# (pipeline SageMaker) — ini hanya pelengkap.
# Jalankan di folder yang sama dengan data_A.csv + script 2a (mis. ~/SageMaker/UAS):
# python generate_artifacts.py
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from preprocessing import clean_data, TARGET, FREE_TEXT_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ["ingested", "train", "test", "model", "eval"]:
    os.makedirs(os.path.join(HERE, sub), exist_ok=True)


def p(*parts):
    return os.path.join(HERE, *parts)


# --- Step 1: Ingest ---
df = pd.read_csv(p("data_A.csv"), index_col=0)
df.to_csv(p("ingested", "data_A.csv"), index=False)
print(f"[1] ingested/data_A.csv  ({df.shape[0]} baris)")

# --- Step 2: Preprocess (clean + split) ---
df = clean_data(df)
df = df.drop(columns=[c for c in FREE_TEXT_COLS if c in df.columns])
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])
train_df.to_csv(p("train", "train.csv"), index=False)
test_df.to_csv(p("test", "test.csv"), index=False)
print(f"[2] train/train.csv {train_df.shape} | test/test.csv {test_df.shape}")

# --- Step 3: Train RandomForest ---
X = train_df.drop(columns=[TARGET])
le = LabelEncoder()
y = le.fit_transform(train_df[TARGET])
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([
    ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
    ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
model = Pipeline([("prep", pre), ("model", RandomForestClassifier(
    n_estimators=200, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1))])
model.fit(X, y)
joblib.dump({"pipeline": model, "label_encoder": le, "feature_order": X.columns.tolist(),
             "classes": le.classes_.tolist(), "num_cols": num, "cat_cols": cat,
             "model_name": "RandomForest"}, p("model", "model_credit_score.joblib"))
print("[3] model/model_credit_score.joblib")

# --- Step 4: Evaluate ---
Xt = test_df.drop(columns=[TARGET])
yt = le.transform(test_df[TARGET])
yp = model.predict(Xt)
macro_f1 = float(f1_score(yt, yp, average="macro"))
acc = float(accuracy_score(yt, yp))
report = {"multiclass_classification_metrics": {
    "macro_f1": {"value": macro_f1, "standard_deviation": "NaN"},
    "accuracy": {"value": acc, "standard_deviation": "NaN"}}}
with open(p("eval", "evaluation.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"[4] eval/evaluation.json  ->  Macro F1 = {macro_f1:.4f} | Accuracy = {acc:.4f}")
print("\nSelesai. Folder ingested/ train/ test/ model/ eval/ sudah terisi.")
