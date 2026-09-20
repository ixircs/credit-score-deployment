# Streamlit UI untuk inferencing model credit score terbaik.
# Form 22 fitur nasabah + 3 preset test case (Good/Standard/Poor) untuk
# memudahkan screenshot tiap kelas. Jalankan: streamlit run app.py
import json

import numpy as np
import pandas as pd
import streamlit as st

from inference import CreditScorePredictor

UNKNOWN = "(tidak diketahui)"   # sentinel kategori kosong -> NaN (diimputasi model)

CLASS_STYLE = {
    "Good": ("Kredit BAIK", "#2ca02c"),
    "Standard": ("Kredit STANDAR", "#ff7f0e"),
    "Poor": ("Kredit BURUK", "#d62728"),
}


@st.cache_resource
def load_predictor():
    return CreditScorePredictor()


@st.cache_data
def load_meta():
    with open("app_metadata.json", encoding="utf-8") as f:
        return json.load(f)


def apply_test_case(meta, name):
    # Isi session_state dari preset test case bernama `name`.
    case = meta["test_cases"][name]
    for col, val in case.items():
        key = f"in_{col}"
        if col in meta["cat_cols"]:
            st.session_state[key] = val if val is not None else UNKNOWN
        else:  # numerik; None -> pakai median
            if val is None:
                val = meta["num_stats"][col]["median"]
            st.session_state[key] = float(val)


def main():
    st.set_page_config(page_title="Credit Score Classifier",
                       layout="wide")
    predictor = load_predictor()
    meta = load_meta()

    st.title("Credit Score Classifier")
    st.caption(
        f"Model: **{predictor.model_name}** (tuned) — Dataset A · "
        "Memprediksi performa kredit nasabah: Good / Standard / Poor."
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
    num_cols = meta["num_cols"]
    grid = st.columns(3)
    for i, col in enumerate(num_cols):
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

    # ---- Prediksi ----
    if st.button("Prediksi Credit Score", type="primary", use_container_width=True):
        record = {}
        for col, val in inputs.items():
            if col in cat_cols and val == UNKNOWN:
                record[col] = np.nan          # kosong -> diimputasi model
            else:
                record[col] = val
        result = predictor.predict(record)

        label = result["label"]
        text, color = CLASS_STYLE.get(label, (label, "#333"))
        st.markdown(
            f"<div style='padding:1rem;border-radius:.5rem;background:{color}22;"
            f"border-left:6px solid {color}'>"
            f"<h2 style='margin:0;color:{color}'>{text}</h2>"
            f"<p style='margin:.2rem 0 0'>Prediksi <b>Credit_Score = {label}</b></p></div>",
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("**Probabilitas per kelas**")
        proba = result["probabilities"]
        pdf = (pd.DataFrame({"Kelas": list(proba), "Probabilitas": list(proba.values())})
                 .set_index("Kelas"))
        st.bar_chart(pdf)
        cols = st.columns(len(proba))
        for c, (cls, p) in zip(cols, proba.items()):
            c.metric(cls, f"{p*100:.1f}%")


if __name__ == "__main__":
    main()
