#!/bin/bash
set -e

SERVER_USER="root"
SERVER_HOST="${SERVER_HOST:-192.168.1.7}"
SERVER_PASS="${SERVER_PASS:-openwifi}"
SERVER_DIR="/mnt/storage/www/web/smartthesis"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo "🚀 Mendeploy SmartThesis UMPO ke Server ($SERVER_HOST)"
echo "📂 Direktori Lokal: $SCRIPT_DIR"
echo "📂 Direktori Server: $SERVER_DIR"
echo "================================================================"

# 1. Pastikan folder tujuan di server sudah ada
echo "📁 [1/3] Menyiapkan direktori server..."
sshpass -p "$SERVER_PASS" ssh -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_HOST" "mkdir -p $SERVER_DIR"

# 2. Sinkronisasi File ke Server
echo "📤 [2/3] Menyinkronkan file perubahan ke $SERVER_USER@$SERVER_HOST:$SERVER_DIR..."
RSYNC_EXCLUDES=(
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude '.git'
  --exclude 'venv'
)

sshpass -p "$SERVER_PASS" rsync -avz --progress "${RSYNC_EXCLUDES[@]}" "$SCRIPT_DIR/" "$SERVER_USER@$SERVER_HOST:$SERVER_DIR/"

# 3. Reload Service & Restart di Server (Systemd + Venv Production)
echo "🔄 [3/3] Memperbarui dependensi & Merestart SmartThesis Service..."
REMOTE_COMMANDS="
  cd $SERVER_DIR
  
  if [ ! -d \"venv\" ]; then
    echo 'Membuat venv Python...'
    python3 -m venv venv
  fi

  echo 'Memeriksa dependensi...'
  venv/bin/pip install -q -r requirements.txt || true
  
  echo 'Merestart systemd service smartthesis...'
  systemctl restart smartthesis
  sleep 3
  
  echo 'Status Service:'
  systemctl status smartthesis --no-pager -n 5
"

sshpass -p "$SERVER_PASS" ssh -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_HOST" "$REMOTE_COMMANDS"

echo "================================================================"
echo "✅ Deployment Sukses! SmartThesis UMPO Aktif di Server."
echo "🌐 Akses Web Server: http://$SERVER_HOST:8501"
echo "================================================================"
