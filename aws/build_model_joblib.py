# build_model_joblib.py
# =====================
# Konversi artefak point 1 `model_credit_score.pkl` -> `model_credit_score.joblib`
# untuk dikemas ke SageMaker (model.tar.gz).
# Isi bundle TIDAK diubah (pipeline + label_encoder + metadata identik); hanya
# nama/format file yang diselaraskan dengan yang dibaca inference.py:model_fn.
# Jalankan (di env training agar versi serialisasi konsisten, sklearn 1.8.0):
# conda run -n tugas_md python aws/build_model_joblib.py
import os

import joblib

SRC = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model_credit_score.pkl")
DST = os.path.join(os.path.dirname(__file__), "model_credit_score.joblib")


def main():
    bundle = joblib.load(SRC)
    # Sanity check: pastikan kunci yang dipakai inference.py ada.
    for k in ("pipeline", "label_encoder", "feature_order", "classes"):
        assert k in bundle, f"Kunci '{k}' tidak ada di bundle!"
    joblib.dump(bundle, DST, compress=3)
    size_mb = round(os.path.getsize(DST) / 1e6, 2)
    print(f"OK -> {DST} ({size_mb} MB)")
    print(f"     model  : {bundle.get('model_name')}")
    print(f"     classes: {bundle.get('classes')}")
    print(f"     fitur  : {len(bundle.get('feature_order', []))} kolom")


if __name__ == "__main__":
    main()
