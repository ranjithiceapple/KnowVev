# V2 Deployment Guide

## Quick Start

```bash
cd deployment

# Deploy V2 (runs alongside V1)
chmod +x *.sh
./deploy_v2.sh

# Test V2
curl http://localhost:8001/v2/jobs

# Switch to V2 (production traffic)
./switch_to_v2.sh

# Rollback if issues
./rollback_to_v1.sh
```

## Services

- **V1**: Port 8000, systemd service `knowvec`
- **V2**: Port 8001, systemd service `knowvec-v2`

## Commands

```bash
# Status
sudo systemctl status knowvec      # V1
sudo systemctl status knowvec-v2   # V2

# Logs
sudo journalctl -u knowvec -f      # V1
sudo journalctl -u knowvec-v2 -f   # V2

# Restart
sudo systemctl restart knowvec-v2
```

## Nginx Routing

Copy nginx config:
```bash
sudo cp nginx.conf /etc/nginx/sites-available/knowvec
sudo ln -s /etc/nginx/sites-available/knowvec /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## Testing V2

```bash
# Upload document
curl -X POST http://localhost:8001/v2/ingest \
  -F "file=@test.pdf"

# Get job status
curl http://localhost:8001/v2/jobs/{job_id}

# Search
curl "http://localhost:8001/v2/search?query=test"
```

## Rollback Process

1. Run `./rollback_to_v1.sh`
2. Verify V1: `curl http://localhost:8000/health`
3. Check logs: `sudo journalctl -u knowvec -f`
4. Optional: Stop V2 service

## Gradual Migration

1. Deploy V2: `./deploy_v2.sh`
2. Test V2 endpoints
3. Route subset of traffic to V2 (nginx weighted)
4. Monitor metrics
5. Full switch: `./switch_to_v2.sh`
6. Keep V1 running as backup for 1 week
