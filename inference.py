import joblib
import numpy as np
import pandas as pd

from pre_processing import clean_data

MODEL_PATH = "artifacts/model_credit_score.pkl"


class CreditScorePredictor:
    # Muat artefak model (.pkl) dan lakukan prediksi credit score.

    def __init__(self, model_path=MODEL_PATH):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.label_encoder = bundle["label_encoder"]
        self.feature_order = bundle["feature_order"]
        self.classes = bundle["classes"]
        self.model_name = bundle.get("model_name", "model")
        self.num_cols = bundle.get("num_cols", [])
        self.cat_cols = bundle.get("cat_cols", [])

    def _prepare(self, data):
        # Terima dict (1 nasabah) atau DataFrame (banyak) -> X siap-prediksi.
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("data harus dict atau pandas.DataFrame")

        # Bersihkan pakai fungsi yang sama seperti waktu training (konsisten).
        df = clean_data(df)

        # Pastikan semua kolom fitur ada; kolom hilang -> NaN (diimputasi pipeline).
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = np.nan
        return df[self.feature_order]

    def predict(self, data):
        # Prediksi label + probabilitas. dict -> 1 hasil; DataFrame -> list hasil.
        X = self._prepare(data)
        idx = self.pipeline.predict(X)
        labels = self.label_encoder.inverse_transform(idx)
        proba = self.pipeline.predict_proba(X)

        results = []
        for i in range(len(X)):
            results.append({
                "label": str(labels[i]),
                "probabilities": {
                    cls: round(float(p), 4)
                    for cls, p in zip(self.classes, proba[i])
                },
            })
        return results[0] if isinstance(data, dict) else results


def _smoke_test():
    # Uji cepat: prediksi pada input minimal (kolom kosong -> imputasi).
    predictor = CreditScorePredictor()
    print("Model :", predictor.model_name)
    print("Kelas :", predictor.classes)

    sample = {col: np.nan for col in predictor.feature_order}
    sample.update({
        "Annual_Income": 50000, "Monthly_Inhand_Salary": 4000,
        "Outstanding_Debt": 800, "Interest_Rate": 8,
        "Num_of_Delayed_Payment": 2, "Credit_Mix": "Good",
    })
    result = predictor.predict(sample)
    print("Prediksi:", result["label"])
    print("Proba   :", result["probabilities"])


if __name__ == "__main__":
    _smoke_test()
