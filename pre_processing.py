import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET_COL = "Credit_Score"

# Kolom identitas: tidak prediktif + berisiko leakage -> dibuang.
ID_COLS = ["ID", "Customer_ID", "Name", "SSN"]

# Kolom numerik yang sering kotor (ketempelan '_', spasi) -> perlu dibersihkan.
DIRTY_NUMERIC_COLS = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt",
    "Amount_invested_monthly", "Monthly_Balance",
]

# Kolom teks bebas multi-nilai yang sulit di-encode -> dibuang dari fitur.
FREE_TEXT_COLS = ["Type_of_Loan"]


def _credit_history_to_months(value):
    # Ubah 'NN Years and MM Months' -> total bulan. NaN tetap NaN.
    if pd.isna(value):
        return np.nan
    years = re.search(r"(\d+)\s*Year", str(value))
    months = re.search(r"(\d+)\s*Month", str(value))
    y = int(years.group(1)) if years else 0
    m = int(months.group(1)) if months else 0
    return y * 12 + m


def clean_data(df):
    # Bersihkan data credit score mentah jadi siap-model.
    # NaN sengaja tidak diisi di sini; imputasi dilakukan di ColumnTransformer
    # agar bebas leakage.
    df = df.copy()

    # Buang kolom identitas.
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")

    # Angka kotor -> numerik.
    for col in DIRTY_NUMERIC_COLS:
        if col not in df.columns:
            continue
        df[col] = (
            df[col].astype(str)
                   .str.replace("_", "", regex=False)
                   .str.strip()
                   .replace({"": np.nan, "nan": np.nan})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Nilai mustahil -> NaN.
    if "Age" in df:
        df.loc[(df["Age"] < 14) | (df["Age"] > 100), "Age"] = np.nan
    if "Num_of_Loan" in df:
        df.loc[df["Num_of_Loan"] < 0, "Num_of_Loan"] = np.nan
    if "Num_of_Delayed_Payment" in df:
        df.loc[df["Num_of_Delayed_Payment"] < 0, "Num_of_Delayed_Payment"] = np.nan
    if "Num_Bank_Accounts" in df:
        df.loc[(df["Num_Bank_Accounts"] < 0) | (df["Num_Bank_Accounts"] > 20),
               "Num_Bank_Accounts"] = np.nan
    if "Num_Credit_Card" in df:
        df.loc[df["Num_Credit_Card"] > 15, "Num_Credit_Card"] = np.nan
    if "Interest_Rate" in df:
        df.loc[df["Interest_Rate"] > 50, "Interest_Rate"] = np.nan
    if "Num_Credit_Inquiries" in df:
        df.loc[df["Num_Credit_Inquiries"] > 30, "Num_Credit_Inquiries"] = np.nan

    # Placeholder palsu -> NaN.
    if "Occupation" in df:
        df["Occupation"] = df["Occupation"].replace("_______", np.nan)
    if "Credit_Mix" in df:
        df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
    if "Payment_of_Min_Amount" in df:
        df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)

    # Credit_History_Age -> total bulan.
    if "Credit_History_Age" in df:
        df["Credit_History_Age"] = df["Credit_History_Age"].apply(_credit_history_to_months)

    return df


class DataPreprocessor:
    # Class preprocessing: cleaning, pisah fitur/target, dan bikin ColumnTransformer
    # (imputasi + scaling numerik + one-hot kategorikal). Transformer di-fit di dalam
    # Pipeline bersama model supaya bebas leakage.

    def __init__(self):
        self.num_cols = None
        self.cat_cols = None

    def clean(self, df):
        return clean_data(df)

    def split_features_target(self, df_clean):
        # Pisah X dan y. Buang target + kolom teks bebas dari fitur.
        drop_cols = [c for c in ([TARGET_COL] + FREE_TEXT_COLS) if c in df_clean.columns]
        X = df_clean.drop(columns=drop_cols)
        y = df_clean[TARGET_COL] if TARGET_COL in df_clean.columns else None
        return X, y

    def build_transformer(self, X):
        # Bikin ColumnTransformer berdasarkan tipe kolom X.
        self.num_cols = X.select_dtypes(include="number").columns.tolist()
        self.cat_cols = X.select_dtypes(exclude="number").columns.tolist()

        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        return ColumnTransformer([
            ("num", numeric_pipe, self.num_cols),
            ("cat", categorical_pipe, self.cat_cols),
        ])
