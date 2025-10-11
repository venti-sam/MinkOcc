#!/usr/bin/env bash
set -euo pipefail


SVC=${1:-minkocc-dev}


if docker compose version >/dev/null 2>&1; then
DC="docker compose"
else
DC="docker-compose"
fi


cd "$(dirname "$0")"
$DC restart "$SVC"
$DC ps