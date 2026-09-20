# SageMaker inference entry point untuk model Credit Score Classification (Dataset A).
# Melayani model sebagai real-time endpoint dengan 4 fungsi kontrak SageMaker:
#   model_fn   - muat model dari disk (sekali per container)
#   input_fn   - parse request body -> DataFrame (per request)
#   predict_fn - jalankan prediksi (per request)
#   output_fn  - serialisasi respons (per request)
# clean_data dipakai sama seperti training supaya input diproses konsisten
# (anti train/serve skew). pre_processing.py disertakan di source_dir agar bisa di-import.
import json
import os

import joblib
import numpy as np
import pandas as pd

from pre_processing import clean_data

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"

MODEL_FILENAME = "model_credit_score.joblib"


def model_fn(model_dir: str):
    # Muat bundle artefak (pipeline + label_encoder + metadata) dari model_dir.
    return joblib.load(os.path.join(model_dir, MODEL_FILENAME))


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    # Parse request body menjadi DataFrame mentah (belum dibersihkan).
    # JSON (disarankan): {"instances": [ {kolom: nilai, ...}, ... ]}
    # Tiap instance = objek fitur nasabah (numerik + kategorikal).
    # Nilai boleh null/hilang -> nanti diimputasi oleh pipeline.
    # CSV: baris berheader, kolom sesuai nama fitur.
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        return pd.DataFrame(instances)

    if request_content_type == CSV_CONTENT_TYPE:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        from io import StringIO
        return pd.read_csv(StringIO(request_body))

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: pd.DataFrame, bundle: dict) -> dict:
    # Jalankan inferensi. Alur SAMA seperti api.py (CreditScorePredictor):
    # raw -> clean_data -> susun feature_order (kolom hilang -> NaN)
    # -> pipeline.predict / predict_proba -> label via inverse_transform.
    pipeline = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]
    feature_order = bundle["feature_order"]
    classes = bundle["classes"]

    df = clean_data(input_data)
    for col in feature_order:
        if col not in df.columns:
            df[col] = np.nan
    X = df[feature_order]

    idx = pipeline.predict(X)
    labels = label_encoder.inverse_transform(idx)
    proba = pipeline.predict_proba(X)

    return {
        "labels": [str(x) for x in labels],
        "predictions": [int(i) for i in idx],
        "probabilities": proba.tolist(),
        "classes": list(classes),
    }


def output_fn(prediction: dict, accept_content_type: str):
    # Serialisasi dict prediksi ke body respons (JSON).
    if accept_content_type in (JSON_CONTENT_TYPE, "*/*", None):
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
