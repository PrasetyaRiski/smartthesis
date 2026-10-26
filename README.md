# 🏛️ SmartThesis UMPO

**Enterprise Academic Research Discovery & Hybrid RAG-Lite Thesis Recommendation System**  
*Universitas Muhammadiyah Ponorogo | Program Studi Teknik Informatika*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-smartthesis.prasetyarizki.my.id-0284C7?style=for-the-badge&logo=cloudflare)](https://smartthesis.prasetyarizki.my.id)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald?style=for-the-badge)](LICENSE)

---

## 📌 Ringkasan Sistem

**SmartThesis UMPO** adalah platform pencarian dan rekomendasi topik skripsi/penelitian lintas program studi berbasis kecerdasan buatan (*Artificial Intelligence*). Sistem ini dirancang untuk mengatasi fenomena *"researcher's block"* pada mahasiswa tingkat akhir melalui pendekatan **Information Retrieval (IR)** berbasis **TF-IDF N-Gram** dan **Cosine Similarity**, yang dipadukan secara hibrida dengan **Generative AI / Large Language Model (Google Gemini)** sebagai asisten akademik interaktif (*Hybrid RAG-Lite Advisor*).

Dataset sistem memuat repositori naskah skripsi riil dari **10 Program Studi di Universitas Muhammadiyah Ponorogo** (*Teknik Informatika, Teknik Mesin, Teknik Elektro, Manajemen, Akuntansi, Ilmu Hukum, Ilmu Komunikasi, Pendidikan Guru PAUD, Pendidikan Bahasa Inggris, dan Keperawatan*).

---

## ✨ Fitur Unggulan

1. **🎯 Multi-Prodi Smart Matching Engine**
   - Pencarian semantik judul skripsi berdasarkan kata kunci minat, abstrak, atau metodologi penelitian.
   - Perhitungan kesamaan dokumen menggunakan matriks **TF-IDF N-Gram (1,2)** dan **Cosine Similarity Vector Space**.
   - Filter ketat berdasarkan Program Studi, batas minimum skor relevansi, dan rentang tahun publikasi.

2. **🤖 Asisten Riset Cerdas (Hybrid RAG-Lite LLM Advisor)**
   - Terintegrasi langsung dengan **Google Gemini AI**.
   - Analisis *Research Gap* (celah kebaruan penelitian) otomatis dari dokumen rujukan yang ditemukan.
   - Formulasi rumusan masalah, rekomendasi variabel penelitian, dan usulan kerangka metodologi yang kontekstual.

3. **📊 Visual Analytics & Topic Clustering**
   - Distribusi frekuensi kata kunci populer (*academic word distribution*).
   - Analisis topik tren penelitian lintas fakultas.

4. **💎 Desain Antarmuka Academic Editorial (Streamlit Enterprise)**
   - UI responsif, elegan, berstandar aksesibilitas tinggi dengan tipografi proporsional.
   - Mode tema otomatis (Dark / Light).

---

## 📁 Struktur Direktori

```text
smartthesis/
├── .streamlit/
│   ├── config.toml                     # Konfigurasi tema visual & server Streamlit
│   └── secrets.toml.example            # Template API Key Google Gemini
├── models/
│   ├── thesis_database_all_prodi.csv   # Database 200 skripsi riil lintas prodi UMPO
│   └── thesis_tfidf_model.pkl          # Serialisasi matriks model TF-IDF Vectorizer
├── app.py                              # Core aplikasi web Streamlit
├── requirements.txt                    # Dependensi pustaka Python
├── Dockerfile                          # Kontainerisasi Docker multi-arsitektur
├── docker-compose.yml                  # Orkestrasi Docker satu-perintah
├── deploy.sh                           # Script otomatisasi deployment VPS
├── deploy_vps_nginx.md                 # Panduan deployment VPS & Nginx Reverse Proxy
├── LICENSE                             # Lisensi MIT
└── README.md                           # Dokumentasi proyek
```

---

## 🚀 Panduan Menjalankan di Komputer Lokal

### 1. Kloning Repositori
```bash
git clone https://github.com/PrasetyaRiski/smartthesis.git
cd smartthesis
```

### 2. Buat Virtual Environment & Install Dependensi
```bash
python3 -m venv venv
source venv/bin/activate  # Untuk Linux/macOS
# venv\Scripts\activate   # Untuk Windows

pip install -r requirements.txt
```

### 3. Konfigurasi API Key (Opsional untuk Fitur AI RAG)
Salin template konfigurasi rahasia:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Buka `.streamlit/secrets.toml` dan masukkan Google Gemini API Key Anda (dapatkan gratis di [Google AI Studio](https://aistudio.google.com/)).

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```
Buka peramban di `http://localhost:8501`.

---

## 🐳 Menjalankan dengan Docker

```bash
docker compose up -d --build
```
Aplikasi akan langsung aktif di `http://localhost:8501`.

---

## 🌐 Deployment Produksi (Systemd / Cloudflare Tunnel)

Aplikasi dapat dijalankan sebagai service 24/7 di server Linux menggunakan Systemd:

```ini
[Unit]
Description=SmartThesis UMPO Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/mnt/storage/www/web/smartthesis
ExecStart=/mnt/storage/www/web/smartthesis/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always

[Install]
WantedBy=multi-user.target
```

Untuk menghubungkan ke domain publik ber-SSL via Cloudflare Tunnel, cukup arahkan Public Hostname ke `http://localhost:8501` dengan dukungan WebSocket aktif.

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah lisensi **MIT** — lihat berkas [LICENSE](LICENSE) untuk detail selengkapnya.
