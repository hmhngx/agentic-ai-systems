#!/usr/bin/env bash
# Pull and start Qdrant locally.
# Data is persisted to ./qdrant_storage so the collection survives restarts.
# Port 6333: REST API (used by qdrant-client)
# Port 6334: gRPC API (not used here but exposed for completeness)

set -euo pipefail

STORAGE_DIR="$(pwd)/qdrant_storage"
mkdir -p "$STORAGE_DIR"

echo "Starting Qdrant on localhost:6333..."

docker run -d \
  --name qdrant_bench \
  --rm \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$STORAGE_DIR:/qdrant/storage" \
  qdrant/qdrant:latest

# Wait for the REST API to be ready (max 30s)
echo "Waiting for Qdrant to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
    echo "Qdrant is ready."
    exit 0
  fi
  sleep 1
done

echo "ERROR: Qdrant did not start within 30 seconds."
exit 1
