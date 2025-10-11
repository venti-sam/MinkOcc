#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# The local patch file. Assumes it's in the same directory as the script.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SOURCE_PATCH_FILE="${SCRIPT_DIR}/unified_patch.py"

# The destination directory and file that will be handled.
TMP_DIR="/tmp/pytorch3d"
TARGET_FILE="${TMP_DIR}/pytorch3d/renderer/points/pulsar/unified.py"
# ---

# --- Preparation Steps ---
echo "INFO: Step 1/7 - Cleaning up previous build directory..."
# This ensures the git clone will not fail if the directory already exists.
rm -rf "${TMP_DIR}"

echo "INFO: Step 2/7 - Uninstalling any existing PyTorch3D version for a clean install..."
pip uninstall -y pytorch3d

echo "INFO: Step 3/7 - Verifying patch file exists..."
if [ ! -f "${SOURCE_PATCH_FILE}" ]; then
    echo "ERROR: Patch file not found at ${SOURCE_PATCH_FILE}"
    exit 1
fi
echo "Patch file found."

# --- Execution Steps ---
echo "INFO: Step 4/7 - Updating packages and installing git..."
apt-get update && apt-get install -y --no-install-recommends git

echo "INFO: Step 5/7 - Cloning PyTorch3D repository to ${TMP_DIR}..."
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git "${TMP_DIR}"

echo "INFO: Step 6/7 - Applying patch by replacing '${TARGET_FILE}'..."
# This command copies your patch file and renames it to unified.py, overwriting the original.
cp "${SOURCE_PATCH_FILE}" "${TARGET_FILE}"

echo "INFO: Step 7/7 - Building, installing, and cleaning up..."
cd "${TMP_DIR}"
pip install --no-cache-dir .
cd /
rm -rf "${TMP_DIR}"
rm -rf /var/lib/apt/lists/*

echo "SUCCESS: Patched PyTorch3D has been successfully reinstalled."