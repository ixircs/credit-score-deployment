# Tahap 2 SageMaker Pipeline: preprocessing.
# Baca data ingested, bersihkan (clean_data), buang Type_of_Loan, lalu split
# 80/20 (stratified) -> train.csv & test.csv.
# clean_data di-inline di sini (self-contained) supaya script bisa jalan mandiri
# di dalam container ProcessingStep tanpa dependensi file lain.
# Imputasi/scaling TIDAK dilakukan di sini (dikerjakan di dalam Pipeline model saat
# train, supaya bebas leakage).
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Credit_Score"
ID_COLS = ["ID", "Customer_ID", "Name", "SSN"]
DIRTY_NUMERIC_COLS = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt",
    "Amount_invested_monthly", "Monthly_Balance",
]
FREE_TEXT_COLS = ["Type_of_Loan"]


def _credit_history_to_months(value):
    if pd.isna(value):
        return np.nan
    years = re.search(r"(\d+)\s*Year", str(value))
    months = re.search(r"(\d+)\s*Month", str(value))
    y = int(years.group(1)) if years else 0
    m = int(months.group(1)) if months else 0
    return y * 12 + m


def clean_data(df):
    df = df.copy()
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")

    for col in DIRTY_NUMERIC_COLS:
        if col not in df.columns:
            continue
        df[col] = (df[col].astype(str).str.replace("_", "", regex=False).str.strip()
                   .replace({"": np.nan, "nan": np.nan}))
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Age" in df:
        df.loc[(df["Age"] < 14) | (df["Age"] > 100), "Age"] = np.nan
    if "Num_of_Loan" in df:
        df.loc[df["Num_of_Loan"] < 0, "Num_of_Loan"] = np.nan
    if "Num_of_Delayed_Payment" in df:
        df.loc[df["Num_of_Delayed_Payment"] < 0, "Num_of_Delayed_Payment"] = np.nan
    if "Num_Bank_Accounts" in df:
        df.loc[(df["Num_Bank_Accounts"] < 0) | (df["Num_Bank_Accounts"] > 20), "Num_Bank_Accounts"] = np.nan
    if "Num_Credit_Card" in df:
        df.loc[df["Num_Credit_Card"] > 15, "Num_Credit_Card"] = np.nan
    if "Interest_Rate" in df:
        df.loc[df["Interest_Rate"] > 50, "Interest_Rate"] = np.nan
    if "Num_Credit_Inquiries" in df:
        df.loc[df["Num_Credit_Inquiries"] > 30, "Num_Credit_Inquiries"] = np.nan

    if "Occupation" in df:
        df["Occupation"] = df["Occupation"].replace("_______", np.nan)
    if "Credit_Mix" in df:
        df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
    if "Payment_of_Min_Amount" in df:
        df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)

    if "Credit_History_Age" in df:
        df["Credit_History_Age"] = df["Credit_History_Age"].apply(_credit_history_to_months)

    return df


if __name__ == "__main__":
    if os.path.exists("/opt/ml/processing"):
        in_dir = "/opt/ml/processing/ingested"
        out_train = "/opt/ml/processing/train"
        out_test = "/opt/ml/processing/test"
    else:
        base = "/home/ec2-user/SageMaker/credit_pipeline"
        in_dir = f"{base}/ingested"
        out_train = f"{base}/train"
        out_test = f"{base}/test"

    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    df = pd.read_csv(os.path.join(in_dir, "data_A.csv"))
    df = clean_data(df)
    df = df.drop(columns=[c for c in FREE_TEXT_COLS if c in df.columns])

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    train_df.to_csv(os.path.join(out_train, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_test, "test.csv"), index=False)
    print(f"Preprocessing OK: train={train_df.shape}, test={test_df.shape}")
