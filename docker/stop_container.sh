#!/usr/bin/env bash
set -euo pipefail

# This script stops and removes the service container, networks, and volumes
# defined in the docker-compose.yml file.

SVC=${1:-minkocc-dev}

# Detect Compose v2 vs v1
if docker compose version >/dev/null 2>&1; then
  DC="docker compose"
else
  DC="docker-compose"
fi

cd "$(dirname "$0")"

echo "Stopping and removing service '$SVC'..."
# Use `down` to stop and remove containers and networks.
$DC down "$SVC"

echo "Service '$SVC' has been stopped and removed."