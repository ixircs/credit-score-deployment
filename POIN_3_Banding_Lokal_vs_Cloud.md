# Poin 3 — Perbandingan Deployment Lokal vs Cloud (AWS)

Setelah menjalankan deployment model credit score baik secara **lokal** (poin 1c) maupun
di **cloud AWS** (poin 2a & 2b), berikut perbandingan kelebihan dan kekurangan kedua
pendekatan berdasarkan pengalaman saya langsung mengerjakannya.

---

## 1. Deployment Lokal (Streamlit + model dimuat langsung)

Pada versi lokal, aplikasi Streamlit **memuat file model (.pkl) langsung** ke dalam proses
aplikasi, lalu memprediksi di mesin yang sama.

**Kelebihan:**
- **Sederhana & cepat disiapkan** — cukup `streamlit run app.py`, tanpa konfigurasi server, jaringan, atau izin akses.
- **Gratis** — berjalan di komputer sendiri, tidak ada biaya.
- **Bebas memilih library & versi** — saya bisa memakai model terbaik saya, yaitu **XGBoost** dengan scikit-learn versi terbaru (1.8), tanpa batasan apa pun. Model dengan Macro F1 tertinggi (0.7245) bisa langsung dipakai.
- **Latency rendah** — model ada di proses yang sama, tidak ada komunikasi jaringan.
- **Mudah di-debug** — semua berjalan di satu tempat.

**Kekurangan:**
- **Tidak scalable** — terikat pada satu mesin; jika banyak pengguna mengakses bersamaan, aplikasi tidak kuat.
- **Tidak bisa diakses publik** — hanya bisa diakses di komputer saya (localhost), tidak bisa dibagikan ke orang lain.
- **Model & aplikasi menyatu** — kalau ingin memakai model dari aplikasi lain, harus menyalin modelnya; tidak ada pemisahan sebagai layanan.
- **Tidak mencerminkan kondisi produksi** — cocok untuk development, tapi bukan cara sistem nyata melayani banyak permintaan.

---

## 2. Deployment Cloud (SageMaker Endpoint + Streamlit di EC2)

Pada versi cloud, model disimpan sebagai **SageMaker real-time endpoint** (layanan tersendiri),
dan aplikasi Streamlit di **EC2** memanggilnya lewat API (boto3), dengan URL yang dapat diakses publik.

**Kelebihan:**
- **Scalable** — model berdiri sebagai layanan terpisah (endpoint) yang bisa diperbanyak sesuai beban; bisa melayani banyak client sekaligus.
- **Dapat diakses publik** — aplikasi punya URL publik (`http://<IP>:8501`) yang bisa diakses siapa saja, kapan saja.
- **Pemisahan tanggung jawab (separation of concerns)** — model sebagai layanan, aplikasi sebagai frontend. Endpoint yang sama bisa dipakai berbagai aplikasi, bukan hanya Streamlit.
- **Terkelola (managed)** — AWS menangani server, container, dan ketersediaan endpoint.
- **Keamanan lebih baik** — akses ke endpoint diatur lewat IAM role (LabInstanceProfile), tanpa menyimpan kunci di dalam kode.

**Kekurangan:**
- **Ada biaya** — endpoint (ml.m5.large) dan EC2 dikenai biaya per jam selama berjalan.
- **Lebih kompleks untuk disiapkan** — butuh banyak langkah: mengemas model, upload ke S3, membuat endpoint, meluncurkan EC2, mengatur security group dan IAM role.
- **Terikat pada environment container** — ini yang paling saya rasakan. Container SageMaker memakai scikit-learn versi 1.4 dan tidak menyertakan XGBoost. Akibatnya:
  - Model harus **dilatih ulang di scikit-learn 1.4.2** agar cocok dengan container.
  - Saya **tidak bisa memakai XGBoost** (model terbaik saya) karena instalasinya di container gagal; sebagai gantinya saya memakai **RandomForest** yang native di scikit-learn agar deployment andal.
  Jadi lingkungan cloud **memaksa penyesuaian** pada pilihan model dan versi library.
- **Ada cold start & latency jaringan** — permintaan harus melewati jaringan ke endpoint.
- **Manajemen sumber daya** — harus ingat mematikan endpoint & EC2 setelah selesai agar tidak boros biaya.

---

## 3. Tabel Ringkas

| Aspek | Lokal | Cloud (AWS) |
|---|---|---|
| Cara model dipakai | dimuat langsung di aplikasi | endpoint terpisah, dipanggil via API |
| Skalabilitas | terbatas 1 mesin | scalable, banyak client |
| Akses | lokal (localhost) saja | URL publik |
| Biaya | gratis | bayar per jam |
| Kebebasan versi/library | bebas (XGBoost, sklearn 1.8) | ikut container (RandomForest, sklearn 1.4) |
| Kompleksitas setup | rendah | tinggi (S3, endpoint, EC2, IAM, security group) |
| Keamanan akses | — | terkelola via IAM role |
| Cocok untuk | development & eksperimen | production / layanan nyata |

---

## 4. Kesimpulan

Kedua pendekatan punya perannya masing-masing dan **saling melengkapi**, bukan saling
menggantikan:

- **Deployment lokal cocok untuk tahap development dan eksperimen** — cepat, murah, fleksibel,
  dan memungkinkan saya memakai model terbaik (XGBoost) tanpa batasan. Ini ideal saat masih
  membangun dan menguji model.

- **Deployment cloud cocok untuk tahap produksi** — ketika model perlu diakses banyak pengguna,
  tersedia publik, dan berjalan sebagai layanan yang scalable dan terkelola. Konsekuensinya
  adalah biaya, kompleksitas setup, dan keterikatan pada environment container.

Pengalaman paling berkesan bagi saya adalah bagaimana **environment cloud memaksa kompromi
teknis**: saya harus menyesuaikan versi library dan bahkan mengganti algoritma model (dari
XGBoost ke RandomForest) demi keandalan deployment. Ini mengajarkan bahwa dalam deployment
nyata, **kompatibilitas dan keandalan environment sering sama pentingnya dengan akurasi model**.
