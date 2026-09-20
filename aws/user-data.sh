#!/bin/bash
# Bootstrap EC2 (Amazon Linux 2023) untuk hosting Streamlit frontend Credit Score.
# Tempel ke EC2 launch wizard -> Advanced details -> User data.
# Streamlit memanggil SageMaker endpoint via boto3; kredensial dari
# LabInstanceProfile (AWS Academy). Edit variabel di bawah sebelum launch.

set -eu

# -------- EDIT INI -------------------------------------------------------
GIT_REPO="https://github.com/<user>/<repo>.git"   # repo berisi folder aws/
SUBFOLDER="aws"                 # path folder app di dalam repo (kosong = root)
APP_FILE="app_streamlit.py"
ENDPOINT_NAME="credit-score-endpoint"
# -------------------------------------------------------------------------

REGION="us-east-1"
APP_DIR="/opt/credit-app"
VENV_DIR="/opt/streamlit-venv"

if [ -z "$SUBFOLDER" ]; then
  APP_PATH="$APP_DIR"
else
  APP_PATH="$APP_DIR/$SUBFOLDER"
fi

dnf update -y
dnf install -y python3 python3-pip git

git clone "$GIT_REPO" "$APP_DIR"
chown -R ec2-user:ec2-user "$APP_DIR"

if [ ! -f "$APP_PATH/$APP_FILE" ]; then
  echo "FATAL: $APP_PATH/$APP_FILE tidak ditemukan."
  find "$APP_DIR" -maxdepth 4 -type f | head -40
  exit 1
fi

# venv agar tidak bentrok dengan Python sistem (rpm-managed).
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
if [ -f "$APP_PATH/requirements_app.txt" ]; then
  "$VENV_DIR/bin/pip" install -r "$APP_PATH/requirements_app.txt"
else
  "$VENV_DIR/bin/pip" install streamlit boto3 pandas numpy
fi

cat >/etc/systemd/system/streamlit.service <<EOF
[Unit]
Description=Streamlit Credit Score App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_PATH
Environment=ENDPOINT_NAME=$ENDPOINT_NAME
Environment=AWS_REGION=$REGION
ExecStart=$VENV_DIR/bin/streamlit run $APP_FILE \\
  --server.address 0.0.0.0 \\
  --server.port 8501 \\
  --server.headless true \\
  --server.enableCORS false \\
  --server.enableXsrfProtection false
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now streamlit.service

sleep 5
if systemctl is-active --quiet streamlit; then
  touch "$APP_DIR/.userdata-success"
  chown ec2-user:ec2-user "$APP_DIR/.userdata-success"
else
  echo "FATAL: service streamlit gagal start."
  journalctl -u streamlit -n 30 --no-pager || true
  exit 1
fi
