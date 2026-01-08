#!/bin/bash
# Deploy V2 alongside V1

set -e

echo "=== Deploying KnowVec V2 ==="

# Copy service file
sudo cp knowvec-v2.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Start V2 (runs on port 8008)
sudo systemctl start knowvec-v2
sudo systemctl enable knowvec-v2

echo "V2 deployed on port 8008"
echo "V1 still running on port 8007"
echo ""
echo "Test V2: curl http://localhost:8008/v2/jobs"
echo "Switch to V2: ./switch_to_v2.sh"
echo "Rollback to V1: ./rollback_to_v1.sh"
