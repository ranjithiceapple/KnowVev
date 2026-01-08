#!/bin/bash
# Rollback to V1

set -e

echo "=== Rolling back to V1 ==="

# Revert nginx
if [ -f /etc/nginx/sites-available/knowvec ]; then
    sudo sed -i 's/localhost:8008/localhost:8007/' /etc/nginx/sites-available/knowvec
    sudo nginx -t && sudo systemctl reload nginx
    echo "Nginx reverted to V1 (port 8007)"
fi

# Revert env vars
echo "KNOWVEC_VERSION=v1" | sudo tee /etc/environment.d/knowvec.conf
echo "KNOWVEC_PORT=8007" | sudo tee -a /etc/environment.d/knowvec.conf

# Optional: Stop V2
read -p "Stop V2 service? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo systemctl stop knowvec-v2
    sudo systemctl disable knowvec-v2
    echo "V2 service stopped"
fi

echo "✓ Rolled back to V1"
echo "Monitor V1: sudo journalctl -u knowvec -f"
