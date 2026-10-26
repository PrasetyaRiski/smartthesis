# PANDUAN DEPLOYMENT SERVER PRODUKSI (VPS UBUNTU / NGINX / DOCKER)

## 1. Menjalankan dengan Docker
```bash
cd 03_implementation
docker build -t smartthesis-umpo:latest .
docker run -d -p 8501:8501 -e GEMINI_API_KEY="AIzaSy_Key_Anda" --name smartthesis smartthesis-umpo:latest
```

## 2. Menjalankan dengan Systemd Service (Jalan Otomatis 24 Jam di VPS)
Buat file `/etc/systemd/system/smartthesis.service`:
```ini
[Unit]
Description=SmartThesis UMPO Streamlit Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/smartthesis/03_implementation
Environment="GEMINI_API_KEY=AIzaSy_Key_Anda"
ExecStart=/usr/local/bin/streamlit run app.py --server.port=8501 --server.address=127.0.0.1 --server.headless=true
Restart=always

[Install]
WantedBy=multi-user.target
```
