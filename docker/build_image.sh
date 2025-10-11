#!/usr/bin/env bash
set -euo pipefail

# --- Initialize variables ---
SVC=""
BUILD_ARGS=() # Use an array for build arguments

# --- Argument Parsing ---
# Loop through all passed arguments to find flags and the service name.
for arg in "$@"; do
  if [[ "$arg" == "--no-cache" ]]; then
    BUILD_ARGS+=("--no-cache")
  else
    # Assume any other argument is the service name.
    # If multiple are given, the last one will be used.
    SVC="$arg"
  fi
done

# --- Set Defaults ---
# If SVC is still empty after the loop, use the default.
SVC=${SVC:-minkocc-dev}

# --- Detect Compose v2 vs v1 ---
if docker compose version >/dev/null 2>&1; then
  DC="docker compose"
else
  DC="docker-compose"
fi

# --- Build Command ---
cd "$(dirname "$0")"

echo "INFO: Building service '$SVC'..."
if [ ${#BUILD_ARGS[@]} -gt 0 ]; then
    echo "INFO: Using options: ${BUILD_ARGS[*]}"
fi
echo "---"

# Execute the build. The array expansion "${BUILD_ARGS[@]}" correctly handles
# the case where no --no-cache flag is provided (it expands to nothing).
$DC build "${BUILD_ARGS[@]}" "$SVC"

echo "---"
echo "✅ Build complete."