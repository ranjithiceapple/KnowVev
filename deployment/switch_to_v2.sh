#!/bin/bash
# Switch traffic to V2

set -e

echo "=== Switching to V2 ==="

# Update nginx upstream (if using nginx)
if [ -f /etc/nginx/sites-available/knowvec ]; then
    sudo sed -i 's/localhost:8007/localhost:8008/' /etc/nginx/sites-available/knowvec
    sudo nginx -t && sudo systemctl reload nginx
    echo "Nginx updated to route to V2 (port 8008)"
fi

# Or update environment variable
echo "KNOWVEC_VERSION=v2" | sudo tee /etc/environment.d/knowvec.conf
echo "KNOWVEC_PORT=8008" | sudo tee -a /etc/environment.d/knowvec.conf

echo "✓ Switched to V2"
echo "Monitor logs: sudo journalctl -u knowvec-v2 -f"
