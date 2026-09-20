# Tahap 3 SageMaker Pipeline: training.
# Baca train.csv, bangun Pipeline(preprocessing + RandomForest), fit, simpan bundle
# model ke SM_MODEL_DIR (nanti dikemas jadi model.tar.gz oleh SageMaker).
# RandomForest dipakai karena native scikit-learn -> mulus di container SKLearn
# SageMaker tanpa perlu install library tambahan. Preprocessing (imputasi/scaling/
# one-hot) dibungkus di dalam Pipeline supaya di-fit hanya pada data latih (anti-leakage).
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

TARGET = "Credit_Score"

if __name__ == "__main__":
    train_dir = os.environ.get("SM_CHANNEL_TRAIN",
                               "/home/ec2-user/SageMaker/credit_pipeline/train")
    model_dir = os.environ.get("SM_MODEL_DIR",
                               "/home/ec2-user/SageMaker/credit_pipeline/model")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(train_dir, "train.csv"))
    X = df.drop(columns=[TARGET])
    y_raw = df[TARGET]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ])
    model.fit(X, y)

    bundle = {
        "pipeline": model,
        "label_encoder": le,
        "feature_order": X.columns.tolist(),
        "classes": le.classes_.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "model_name": "RandomForest",
    }
    joblib.dump(bundle, os.path.join(model_dir, "model_credit_score.joblib"))
    print(f"Training OK: {len(num_cols)} num + {len(cat_cols)} cat fitur, "
          f"kelas={le.classes_.tolist()}")
