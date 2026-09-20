# Tahap 1 SageMaker Pipeline: ingestion.
# Baca data mentah data_A.csv, simpan ke folder 'ingested'.
# Path menyesuaikan: di dalam container SageMaker vs test manual di notebook.
import os
import pandas as pd


def ingest_data():
    container_input = "/opt/ml/processing/input"
    if os.path.exists(container_input):
        input_dir = container_input
        output_dir = "/opt/ml/processing/ingested"
    else:
        base = "/home/ec2-user/SageMaker/credit_pipeline"
        input_dir = base
        output_dir = os.path.join(base, "ingested")

    os.makedirs(output_dir, exist_ok=True)
    input_file = os.path.join(input_dir, "data_A.csv")

    if os.path.exists(input_file):
        df = pd.read_csv(input_file, index_col=0)   # kolom pertama = nomor urut
        output_file = os.path.join(output_dir, "data_A.csv")
        df.to_csv(output_file, index=False)
        print(f"Ingestion OK: {df.shape[0]} baris -> {output_file}")
    else:
        print(f"ERROR: {input_file} tidak ditemukan.")
        print("Isi folder:", os.listdir(input_dir))


if __name__ == "__main__":
    ingest_data()
