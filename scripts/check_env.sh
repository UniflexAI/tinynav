#!/bin/bash
set -euo pipefail

# === Check if Docker is installed ===
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    echo "👉 You can install it with the following command:"
    echo "   sudo apt-get update && sudo apt-get install -y docker.io"
    exit 1
else
    echo "✅ Docker is installed."
fi

# === Check if Docker daemon is running and accessible ===
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running or you don't have permission to access it."
    echo "👉 Try running: sudo systemctl start docker"
    echo "   Or ensure your user is in the 'docker' group."
    exit 1
else
    echo "✅ Docker daemon is running and accessible."
fi

# === Check if 'nvidia' runtime is available ===
if docker info | grep -i 'Runtimes' | grep -q 'nvidia'; then
    echo "✅ NVIDIA runtime is available in Docker."
else
    echo "❌ NVIDIA runtime is NOT available in Docker."
    echo "👉 You can install it with the following command:"
    echo "   sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    echo "   Then verify with: docker info | grep Runtimes"
    exit 1
fi

# === Check if git-lfs is installed ===
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS is not installed."
    echo "👉 You can install it with the following command:"
    echo "   sudo apt-get update && sudo apt-get install -y git-lfs && git lfs install"
    exit 1
else
    echo "✅ Git LFS is installed."
fi

# === Architecture-specific devcontainer patch ===
ARCH="$(uname -m)"
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" || "$ARCH" == arm* ]]; then
    if [[ -f .devcontainer/devcontainer.json ]]; then
        sed -i '12s/.*/    "--runtime", "nvidia",/' .devcontainer/devcontainer.json
        echo "✅ devcontainer.json patched for your ARM platform."
    else
        echo "⚠️  ARM platform detected but devcontainer.json not found in current directory."
    fi
else
    echo "✅ devcontainer.json patched for your x86 platform."
fi

