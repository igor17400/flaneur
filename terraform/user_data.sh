#!/bin/bash
set -euo pipefail
exec > /var/log/derive-setup.log 2>&1

echo "=== Derive setup starting ==="

# System packages
apt-get update -y
apt-get install -y git curl nginx

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# Clone repo
cd /opt
git clone --branch ${repo_branch} ${repo_url} flaneur
cd flaneur

# Install Python + dependencies
uv python install 3.12
uv sync --no-dev

# Generate predictions (needed for the platform)
echo "=== Generating predictions ==="
uv run python src/infer.py --all || echo "Warning: infer.py failed (may need checkpoints)"

# Environment file
cat > /opt/flaneur/.env <<'ENVEOF'
MISTRAL_API_KEY=${mistral_api_key}
ENVEOF

# Systemd service for derive
cat > /etc/systemd/system/derive.service <<'EOF'
[Unit]
Description=Derive visualization server
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/flaneur
ExecStart=/root/.local/bin/uv run python derive/server.py
Restart=always
RestartSec=5
Environment=PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable derive
systemctl start derive

# Nginx reverse proxy: port 80 → 8765
cat > /etc/nginx/sites-available/derive <<'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # SSE support for /api/explain
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }
}
EOF

rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/derive /etc/nginx/sites-enabled/derive
nginx -t && systemctl restart nginx

echo "=== Derive setup complete ==="
