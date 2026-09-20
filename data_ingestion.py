import os
import pandas as pd

RAW_PATH = "data_A.csv"
INGESTED_DIR = "ingested"


def ingest_data(path=RAW_PATH, out_dir=INGESTED_DIR):
    #Baca data mentah, simpan salinan ke folder ingested, kembalikan DataFrame.
    df = pd.read_csv(path, index_col=0)   # kolom pertama cuma nomor urut

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, os.path.basename(path))
    df.to_csv(out_file, index=False)

    print(f"Data ingested: {path} -> {df.shape[0]} baris x {df.shape[1]} kolom")
    return df


if __name__ == "__main__":
    ingest_data()
