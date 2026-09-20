# Deploy AWS — Credit Score Classification (rubrik 2a & 2b)

Arsitektur (mengikuti contoh dosen `ML_deploymentAWS`):

```
model_credit_score.joblib ──tar.gz──► S3
                                        │
     SageMaker real-time endpoint ◄── inference.py (model_fn/input_fn/predict_fn/output_fn)
     (SKLearnModel, ml.m5.large, fw 1.4-2 + requirements.txt upgrade sklearn 1.8.0)
                                        │
     Streamlit di EC2 (systemd) ── boto3 sagemaker-runtime ──► endpoint
     (kredensial dari LabInstanceProfile)                      URL publik :8501
```

**Model dipertahankan dari point 1** (XGBoost tuned, Macro F1 0.7245) — TIDAK dilatih
ulang. `requirements.txt` di folder ini meng-upgrade sklearn container ke 1.8.0 saat
boot agar unpickling konsisten.

## Isi folder `aws/`
| File | Peran |
|---|---|
| `model_credit_score.joblib` | Bundle model (pipeline XGBoost + label_encoder + metadata) |
| `inference.py` | Entry point SageMaker (4 fungsi kontrak) |
| `pre_processing.py` | Salinan `clean_data()` — dipakai inference.py di container |
| `requirements.txt` | Deps yang di-install container (sklearn 1.8.0, xgboost 3.2.0, …) |
| `build_model_joblib.py` | Regenerasi `.joblib` dari `../model_credit_score.pkl` (opsional) |
| `deploy_endpoint.py` | Kemas → upload S3 → deploy endpoint → smoke test |
| `cleanup_endpoint.py` | Hapus endpoint + config (hemat biaya) |
| `app_streamlit.py` | Streamlit frontend (panggil endpoint via boto3) |
| `requirements_app.txt` | Deps Streamlit di EC2 (streamlit, boto3, pandas, numpy) |
| `user-data.sh` | Bootstrap EC2 (git clone + venv + systemd service) |

---

## BAGIAN 2a — Deploy model ke SageMaker endpoint

> Semua langkah di **AWS Academy Learner Lab** → **Start Lab** → **AWS** (region `us-east-1`).

1. **Buka SageMaker Notebook** (atau SageMaker Studio Classic / Notebook instance).
   Buat/nyalakan Notebook instance (mis. `ml.t3.medium`), lalu **Open JupyterLab**.

2. **Upload folder `aws/`** ke `~/SageMaker/aws` (drag-drop, atau `git clone` repo-mu).
   Pastikan `model_credit_score.joblib` ikut ter-upload (2.7 MB).

3. **Buka Terminal** di Jupyter (File → New → Terminal) lalu jalankan:
   ```bash
   cd ~/SageMaker/aws
   pip install -q sagemaker            # biasanya sudah ada
   python deploy_endpoint.py
   ```
   Script akan: kemas `model.tar.gz` → upload ke bucket default S3 →
   buat `SKLearnModel` → **deploy `credit-score-endpoint`** (5–8 menit) →
   **smoke test** (harus muncul `"labels": ["Good"]`).

   > 📸 **Screenshot 2a**: output terminal saat endpoint sukses + hasil smoke test,
   > dan halaman **SageMaker → Inference → Endpoints** dengan status **InService**.

4. **Kalau gagal build container** (error install versi di requirements.txt) →
   lihat **Fallback versi** di bawah.

---

## BAGIAN 2b — Streamlit publik di EC2

### Opsi A — via `user-data.sh` (otomatis, disarankan)

1. Push folder `aws/` ke sebuah **repo GitHub publik**. Edit `user-data.sh`:
   set `GIT_REPO` ke URL repo-mu (`SUBFOLDER="aws"`, `APP_FILE="app_streamlit.py"`).

2. **Launch EC2 instance**:
   - AMI: **Amazon Linux 2023**, tipe `t3.small` (cukup).
   - **IAM instance profile**: pilih **LabInstanceProfile** (WAJIB — supaya boto3
     bisa invoke endpoint).
   - **Security Group**: buka inbound **TCP 8501** (Source `0.0.0.0/0`) dan SSH 22.
   - **Advanced details → User data**: tempel isi `user-data.sh`.

3. Tunggu ±3 menit. Buka `http://<EC2-Public-IP>:8501` → aplikasi Streamlit.
   Klik preset **Good/Standard/Poor** → **Prediksi** → hasil datang dari endpoint.

   > 📸 **Screenshot 2b**: browser dengan **URL publik** `http://<IP>:8501` terlihat +
   > hasil prediksi. Ini bukti "public URL".

### Opsi B — manual (kalau tak mau pakai GitHub)

SSH ke EC2, `scp` folder `aws/`, lalu:
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements_app.txt
export ENDPOINT_NAME=credit-score-endpoint AWS_REGION=us-east-1
streamlit run app_streamlit.py --server.address 0.0.0.0 --server.port 8501
```

---

## ⚠️ Setelah selesai (WAJIB — hindari boros kredit)
```bash
cd ~/SageMaker/aws
python cleanup_endpoint.py     # hapus endpoint + config
```
Lalu **Stop** EC2 instance & Notebook instance. Endpoint `ml.m5.large` berbiaya
per jam selama InService.

---

## Fallback versi (kalau container gagal install requirements.txt)
Container fw `1.4-2` memakai Python tertentu; kalau wheel `pandas==3.0.2` /
`numpy==2.4.4` gagal ter-install, longgarkan `requirements.txt` menjadi:
```
scikit-learn==1.8.0
xgboost==3.2.0
pandas
numpy
joblib
```
Kalau `sklearn 1.8.0` sendiri tak bisa (Python container terlalu lama), alternatif
terakhir: latih ulang model di Notebook dengan sklearn 1.4.2 (`pip install
scikit-learn==1.4.2`) memakai `../pipeline.py`, hasilkan `.joblib` baru, dan set
`FRAMEWORK_VERSION="1.4-2"` tanpa override sklearn di requirements.txt.

## Untuk rubrik 3 (banding lokal vs cloud)
- **Lokal** (`app.py` + `api.py`): model `.pkl` dimuat langsung di proses Streamlit —
  latency rendah, tanpa biaya, tapi tak scalable & terikat 1 mesin.
- **Cloud** (endpoint + EC2): model terpisah sebagai layanan (SageMaker) di belakang
  EC2 frontend — bisa di-scale/diakses publik, ada isolasi & instance profile,
  tapi ada biaya per jam & cold start ~beberapa detik.
