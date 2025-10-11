#!/usr/bin/env bash
set -euo pipefail

echo "======================================================"
echo "INFO: Installing SpConv and MinkowskiEngine into the user space..."
echo "======================================================"

# --- Step 1: Install spconv into the user's local site-packages ---
# echo "--> Step 1: Installing SpConv (spconv-cu117)..."
# pip install --no-cache-dir --user spconv-cu117

# --- Step 2: Clone the MinkowskiEngine repository ---
echo "--> Step 2: Cloning MinkowskiEngine repository to /tmp..."
# Ensure the directory is clean before cloning
rm -rf /tmp/MinkowskiEngine
git clone https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine

# --- Step 3: Build and Install into the user's local site-packages ---
# This uses the official method with the critical `--user` flag.
echo "--> Step 3: Compiling and installing MinkowskiEngine with CUDA support..."
cd /tmp/MinkowskiEngine
python setup.py install \
  --user \
  --force_cuda \
  --blas_include_dirs=/usr/include/openblas \
  --blas=openblas

# --- Step 4: Clean up the source code ---
echo "--> Step 4: Cleaning up installation files..."
cd / # Move out of the temp directory before removing it
rm -rf /tmp/MinkowskiEngine

# --- Step 5: Verify the installation and CUDA build ---
# This is the most critical step. It will now find the user-installed package.
echo "--> Step 5: Verifying installation and CUDA build..."
python -c "import MinkowskiEngine as ME; ME.print_diagnostics()"

echo "======================================================"
echo "INFO: Installation complete."
echo "======================================================"