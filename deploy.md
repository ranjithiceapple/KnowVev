# Quick Deploy

## Ports
- V1: 8007
- V2: 8008

## Deploy V2
```bash
cd deployment
chmod +x *.sh
./deploy_v2.sh
```

## Test
```bash
curl http://localhost:8008/v2/jobs
```

## Switch to V2
```bash
./switch_to_v2.sh
```

## Rollback
```bash
./rollback_to_v1.sh
```

## Logs
```bash
sudo journalctl -u knowvec-v2 -f
```

