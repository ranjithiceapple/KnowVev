# Connecting to Existing Qdrant Container

This guide explains how to connect the KnowVec application to an existing Qdrant container instead of starting a new one.

## Prerequisites

- You already have a Qdrant container running
- Qdrant is accessible (ports 6333 and/or 6334)

---

## Option 1: Qdrant on Same Docker Network

If you want KnowVec and Qdrant to communicate via Docker internal network:

### Step 1: Find Qdrant Container Name

```bash
docker ps | grep qdrant
```

Note the container name (e.g., `my-qdrant-container`)

### Step 2: Connect to Same Network

**Option A: Add KnowVec to Qdrant's Network**

```bash
# Find Qdrant's network
docker inspect <qdrant-container-name> | grep NetworkMode

# Update docker-compose.yml - replace 'knowvec-network' with Qdrant's network
```

**Option B: Add Qdrant to KnowVec's Network**

```bash
# After starting KnowVec
docker network connect knowvev_knowvec-network <qdrant-container-name>
```

### Step 3: Update Configuration

In your `.env` file:

```bash
QDRANT_URL=http://<qdrant-container-name>:6333
```

### Step 4: Remove Qdrant from docker-compose.yml

Edit `docker-compose.yml`:

```yaml
services:
  # qdrant:  # COMMENT OUT THE ENTIRE QDRANT SERVICE
  #   image: qdrant/qdrant:latest
  #   ...

  knowvec:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: knowvec-app
    ports:
      - "8007:8007"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    environment:
      # Point to existing Qdrant container
      - QDRANT_URL=http://<qdrant-container-name>:6333
    networks:
      - knowvec-network  # Or use Qdrant's network
    restart: unless-stopped

# Remove 'depends_on' for qdrant since it's external

volumes:
  # Remove or comment out qdrant_storage if not needed
  # qdrant_storage:
  #   driver: local

networks:
  knowvec-network:
    driver: bridge
```

### Step 5: Start KnowVec

```bash
docker-compose up -d knowvec
```

---

## Option 2: Qdrant on Host Network or Different Machine

If Qdrant is running on host network or a different machine:

### Step 1: Find Qdrant URL

**If on same machine:**
```bash
# Qdrant on host network
QDRANT_URL=http://localhost:6333

# Or use host.docker.internal (works on Docker Desktop)
QDRANT_URL=http://host.docker.internal:6333
```

**If on different machine:**
```bash
QDRANT_URL=http://<qdrant-host-ip>:6333
```

### Step 2: Update .env

```bash
QDRANT_URL=http://localhost:6333
# or
QDRANT_URL=http://host.docker.internal:6333
# or
QDRANT_URL=http://192.168.1.100:6333
```

### Step 3: Update docker-compose.yml

If using `localhost` or `host.docker.internal`, add to KnowVec service:

```yaml
knowvec:
  # ... other config ...
  extra_hosts:
    - "host.docker.internal:host-gateway"  # For Docker on Linux
  # ... rest of config ...
```

### Step 4: Remove Qdrant Service

Comment out the entire `qdrant:` service section in docker-compose.yml

### Step 5: Start KnowVec

```bash
docker-compose up -d
```

---

## Option 3: Using External Network

If Qdrant is in a different Docker Compose project:

### Step 1: Make Qdrant Network External

In Qdrant's docker-compose.yml, make network external:

```yaml
networks:
  qdrant-network:
    name: shared-qdrant-network
    driver: bridge
```

Start Qdrant first to create the network.

### Step 2: Use External Network in KnowVec

In KnowVec's docker-compose.yml:

```yaml
services:
  knowvec:
    # ... your config ...
    networks:
      - shared-qdrant-network
    environment:
      - QDRANT_URL=http://<qdrant-container-name>:6333

networks:
  shared-qdrant-network:
    external: true
```

### Step 3: Remove Qdrant Service

Comment out the `qdrant:` service in KnowVec's docker-compose.yml

### Step 4: Start KnowVec

```bash
docker-compose up -d
```

---

## Verification Steps

After connecting, verify the connection:

### 1. Check Qdrant is Accessible

```bash
# From host
curl http://localhost:6333/

# From inside KnowVec container
docker exec -it knowvec-app curl http://<qdrant-url>:6333/
```

### 2. Check KnowVec Logs

```bash
docker-compose logs knowvec | grep -i qdrant
```

Look for:
```
INFO - Qdrant storage initialized
INFO - Connected to Qdrant at http://...
```

### 3. Test API

```bash
# Check collection
curl http://localhost:8007/stats

# Upload a test document
curl -X POST "http://localhost:8007/process" -F "file=@test.pdf"
```

---

## Common Connection Scenarios

### Scenario 1: Qdrant Running Standalone

```bash
# Your existing Qdrant
docker run -d -p 6333:6333 -p 6334:6334 --name my-qdrant qdrant/qdrant

# In .env
QDRANT_URL=http://host.docker.internal:6333

# In docker-compose.yml - remove qdrant service, add:
extra_hosts:
  - "host.docker.internal:host-gateway"
```

### Scenario 2: Qdrant in Different Compose Project

```bash
# Qdrant project
cd /path/to/qdrant-project
docker-compose up -d

# Find network name
docker network ls | grep qdrant

# KnowVec docker-compose.yml
networks:
  qdrant-project-network:
    external: true

# In knowvec service
networks:
  - qdrant-project-network
```

### Scenario 3: Qdrant on Remote Server

```bash
# In .env
QDRANT_URL=http://192.168.1.100:6333

# Or with authentication
QDRANT_URL=http://192.168.1.100:6333
QDRANT_API_KEY=your-api-key-here
```

---

## Troubleshooting

### Connection Refused

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Check if port is accessible
telnet localhost 6333
# or
curl http://localhost:6333/

# Check firewall rules (if on different machine)
sudo ufw status
```

### Network Not Found

```bash
# List all networks
docker network ls

# Inspect network
docker network inspect <network-name>

# Verify container is on the network
docker inspect <container-name> | grep Networks -A 10
```

### DNS Resolution Failed

```bash
# Inside KnowVec container, test DNS
docker exec -it knowvec-app ping <qdrant-container-name>

# If fails, use IP address instead
docker inspect <qdrant-container-name> | grep IPAddress
# Use that IP in QDRANT_URL
```

### Cannot Access from Container

```bash
# Check if containers can communicate
docker exec -it knowvec-app curl http://<qdrant-container-name>:6333/

# If timeout, check network connectivity
docker network inspect <network-name>

# Ensure both containers are on same network
docker network connect <network-name> <container-name>
```

---

## Clean Configuration Example

### Minimal docker-compose.yml (No Qdrant)

```yaml
services:
  knowvec:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: knowvec-app
    ports:
      - "8007:8007"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    environment:
      - QDRANT_URL=http://existing-qdrant:6333
    networks:
      - app-network
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"

networks:
  app-network:
    driver: bridge
```

### .env File

```bash
# Qdrant Configuration (existing instance)
QDRANT_URL=http://existing-qdrant:6333
QDRANT_COLLECTION=knowvec_docs
QDRANT_API_KEY=  # If using authentication

# Embedding Configuration
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIM=768

# Application Configuration
LOG_LEVEL=INFO
CORS_ORIGINS=*
```

---

## Summary

1. **Remove** or comment out the `qdrant` service from docker-compose.yml
2. **Update** `QDRANT_URL` in `.env` to point to existing Qdrant
3. **Ensure** network connectivity (same network, host access, or external network)
4. **Remove** `depends_on: qdrant` if present
5. **Start** KnowVec: `docker-compose up -d`
6. **Verify** connection: `curl http://localhost:8007/stats`

Choose the option that matches your setup and follow the steps above! 🎯
