# Streamlit frontend (di EC2) untuk memanggil SageMaker endpoint.
# Versi cloud dari app.py: alih-alih memuat .pkl lokal, aplikasi ini memanggil
# SageMaker real-time endpoint via boto3 (sagemaker-runtime). Form 22 fitur +
# 3 preset test case, sama seperti app.py.
# Kredensial AWS diambil boto3 dari EC2 instance profile (LabInstanceProfile) saat
# jalan di EC2, atau ~/.aws/credentials saat lokal.
# Env var: ENDPOINT_NAME (default 'credit-score-endpoint'), AWS_REGION (default us-east-1).
# Jalankan: streamlit run app_streamlit.py
import json
import os

import boto3
import numpy as np
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

UNKNOWN = "(tidak diketahui)"   # sentinel kategori kosong -> NaN (diimputasi model)

CLASS_STYLE = {
    "Good": ("Kredit BAIK", "#2ca02c"),
    "Standard": ("Kredit STANDAR", "#ff7f0e"),
    "Poor": ("Kredit BURUK", "#d62728"),
}


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


@st.cache_data
def load_meta():
    with open("app_metadata.json", encoding="utf-8") as f:
        return json.load(f)


def invoke_endpoint(record: dict) -> dict:
    # Kirim 1 record nasabah ke endpoint, kembalikan dict hasil inference.py.
    runtime = get_runtime_client()
    payload = {"instances": [record]}
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(resp["Body"].read().decode("utf-8"))


def apply_test_case(meta, name):
    case = meta["test_cases"][name]
    for col, val in case.items():
        key = f"in_{col}"
        if col in meta["cat_cols"]:
            st.session_state[key] = val if val is not None else UNKNOWN
        else:
            if val is None:
                val = meta["num_stats"][col]["median"]
            st.session_state[key] = float(val)


def main():
    st.set_page_config(page_title="Credit Score Classifier (AWS)",
                       layout="wide")
    meta = load_meta()

    st.title("Credit Score Classifier — SageMaker")
    st.caption(
        f"Model RandomForest — Dataset A · via SageMaker endpoint "
        f"`{ENDPOINT_NAME}` @ {REGION}. Good / Standard / Poor."
    )

    # ---- Preset test case ----
    st.subheader("Uji cepat (test case per kelas)")
    st.write("Klik salah satu untuk mengisi form dengan contoh data nyata, lalu tekan **Prediksi**.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Test Case: Good", use_container_width=True):
        apply_test_case(meta, "Good"); st.rerun()
    if c2.button("Test Case: Standard", use_container_width=True):
        apply_test_case(meta, "Standard"); st.rerun()
    if c3.button("Test Case: Poor", use_container_width=True):
        apply_test_case(meta, "Poor"); st.rerun()

    st.divider()

    # ---- Form input fitur ----
    st.subheader("Data Nasabah")
    inputs = {}

    st.markdown("**Fitur Numerik**")
    grid = st.columns(3)
    for i, col in enumerate(meta["num_cols"]):
        stt = meta["num_stats"][col]
        key = f"in_{col}"
        if key not in st.session_state:
            st.session_state[key] = float(stt["median"])
        inputs[col] = grid[i % 3].number_input(
            col.replace("_", " "), key=key, step=1.0, format="%.2f",
            help=f"min {stt['min']} · median {stt['median']} · max {stt['max']}",
        )

    st.markdown("**Fitur Kategorikal**")
    cat_cols = meta["cat_cols"]
    cgrid = st.columns(len(cat_cols))
    for i, col in enumerate(cat_cols):
        options = meta["cat_options"][col] + [UNKNOWN]
        key = f"in_{col}"
        if key not in st.session_state:
            st.session_state[key] = options[0]
        inputs[col] = cgrid[i].selectbox(col.replace("_", " "), options, key=key)

    st.divider()

    # ---- Prediksi via endpoint ----
    if st.button("Prediksi Credit Score", type="primary", use_container_width=True):
        record = {}
        for col, val in inputs.items():
            if col in cat_cols and val == UNKNOWN:
                record[col] = None            # kosong -> diimputasi model
            else:
                record[col] = val

        try:
            result = invoke_endpoint(record)
        except NoCredentialsError:
            st.error(
                "Kredensial AWS tidak ditemukan. Di EC2: pastikan LabInstanceProfile "
                "terpasang. Lokal: konfigurasi ~/.aws/credentials."
            )
            return
        except ClientError as e:
            st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
            return

        label = result["labels"][0]
        classes = result["classes"]
        proba = dict(zip(classes, result["probabilities"][0]))

        text, color = CLASS_STYLE.get(label, (label, "#333"))
        st.markdown(
            f"<div style='padding:1rem;border-radius:.5rem;background:{color}22;"
            f"border-left:6px solid {color}'>"
            f"<h2 style='margin:0;color:{color}'>{text}</h2>"
            f"<p style='margin:.2rem 0 0'>Prediksi <b>Credit_Score = {label}</b> "
            f"(via SageMaker endpoint)</p></div>",
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("**Probabilitas per kelas**")
        pdf = (pd.DataFrame({"Kelas": list(proba), "Probabilitas": list(proba.values())})
                 .set_index("Kelas"))
        st.bar_chart(pdf)
        cols = st.columns(len(proba))
        for c, (cls, p) in zip(cols, proba.items()):
            c.metric(cls, f"{p*100:.1f}%")


if __name__ == "__main__":
    main()
