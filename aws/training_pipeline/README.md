# 2a — Pipeline Training ML di Cloud (SageMaker Pipeline)

Membangun pipeline training sebagai **SageMaker Pipeline** yang berjalan di AWS
(local mode di dalam Notebook instance). Mengikuti pola contoh dosen
(`Training Pipeline`), diadaptasi untuk data credit score (Dataset A).

**Alur 4 step:** `Ingest → Preprocess → Train → Evaluate`

```
data_A.csv → [Ingest] → ingested/ → [Preprocess] clean+split → train.csv/test.csv
                                          │                              │
                                     [Train] RandomForest          [Evaluate]
                                     → model.tar.gz  ───────────►  Macro F1 + Accuracy
                                                                   → evaluation.json
```

## Catatan model: kenapa RandomForest (bukan XGBoost)?
Pipeline cloud ini pakai **RandomForest** karena native scikit-learn → jalan mulus
di container SKLearn SageMaker **tanpa perlu install xgboost**. Ini juga selaras
dengan kesimpulan eksplorasi di notebook (1a). **Model produksi yang di-deploy (2b)
tetap XGBoost tuned** (Macro F1 0.7245) dari pipeline lokal. Jadi:
- **2a** = mendemokan *mekanisme pipeline training di cloud* (RandomForest).
- **2b** = deploy *model produksi terbaik* (XGBoost) sebagai endpoint + Streamlit.

## File
| File | Peran |
|---|---|
| `data_ingestion.py` | Step 1: baca data_A.csv → ingested/ |
| `preprocessing.py` | Step 2: clean_data + split → train.csv/test.csv (clean_data inline, self-contained) |
| `train.py` | Step 3: Pipeline(ColumnTransformer + RandomForest) → model.joblib |
| `evaluation.py` | Step 4: Macro F1 + accuracy → evaluation.json |
| `pipeline.py` | Orchestrator SageMaker Pipeline (local mode) |

---

## Cara menjalankan (di AWS)

### Prasyarat
1. **Gunakan SageMaker _Notebook instance_ (BUKAN Studio)** — local mode butuh Docker,
   dan Notebook instance sudah punya Docker. (SageMaker → Notebook → Notebook instances
   → Create, mis. `ml.t3.medium` → Open JupyterLab.)
2. Pastikan `sagemaker` SDK terpasang (biasanya sudah): `pip install -q sagemaker`.

### Langkah
1. Upload ke satu folder kerja (nama bebas, mis. **`~/SageMaker/UAS/`**):
   - `data_A.csv` (dataset)
   - `data_ingestion.py`, `preprocessing.py`, `train.py`, `evaluation.py`, `pipeline.py`
   - (folder ini HANYA berisi 6 file itu — jangan campur file point-1 lokal seperti
     `inference.py`/`app.py`/`notebook.ipynb`)
2. Buka **Terminal** (File → New → Terminal), lalu (`cd` ke folder tempat file berada):
   ```bash
   cd ~/SageMaker/UAS
   python pipeline.py
   ```
   `pipeline.py` otomatis memakai folder tempat ia berada, jadi nama folder bebas.
3. Pipeline akan: registrasi definisi → jalan local mode (Docker menarik image SKLearn
   pada run pertama, agak lama) → menjalankan 4 step berurutan. Di akhir tercetak:
   ```
   Status akhir tiap step:
     CreditIngest         -> Succeeded
     CreditPreprocess     -> Succeeded
     CreditTrain          -> Succeeded
     CreditEval           -> Succeeded
   ```
4. Cek hasil evaluasi:
   ```bash
   cat eval/evaluation.json
   ```

### 📸 Screenshot untuk rubrik 2a
- **Terminal** yang menampilkan **4 step `Succeeded`** + isi `evaluation.json` (Macro F1).
- **SageMaker → Pipelines** → pipeline `Credit-Local-Workflow` (definisi terdaftar) —
  bila muncul, screenshot graph/execution-nya juga.
- (Opsional) folder hasil: `ingested/`, `train/`, `test/`, `eval/` terisi.

---

## Troubleshooting
- **`docker: command not found` / local mode gagal** → kamu sedang di SageMaker Studio,
  bukan Notebook instance. Pindah ke Notebook instance.
- **Image pull lama / timeout** → run pertama menarik image container; ulangi `python pipeline.py`.
- **`data_A.csv not found`** → pastikan file ada di `~/SageMaker/credit_pipeline/` (bukan subfolder).
- **Role error** → di Learner Lab, `pipeline.py` otomatis fallback ke `LabRole`.

## Hubungan dengan 2b
Pipeline 2a menghasilkan `model.tar.gz` (RandomForest) di folder `eval/`/output step.
Untuk **deployment (2b)** kita pakai model XGBoost produksi (folder `aws/`), lihat
`aws/README.md`.
