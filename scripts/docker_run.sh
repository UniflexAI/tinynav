#!/bin/bash

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
fi

# === Check if 'nvidia' runtime is available ===
if docker info | grep -i 'Runtimes' | grep -q 'nvidia'; then
    echo "✅ NVIDIA runtime is available in Docker."
else
    echo "❌ NVIDIA runtime is NOT available in Docker."
    echo "👉 You can install it with the following command:"
    echo "   sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker"
    echo "   Then verify with: docker info | grep Runtimes"
    exit 1
fi
# Detect architecture
ARCH=$(uname -m)

# Default Docker arguments
DOCKER_ARGS="-it --rm"

# Use NVIDIA GPU differently depending on platform: https://forums.developer.nvidia.com/t/whats-difference-between-gpus-and-runtime-nvidia-for-the-docker-container/283468/3
if [[ "$ARCH" == "aarch64" ]]; then
    # Jetson (ARM)
    echo "🟢 Detected Jetson (ARM64) platform"
    DOCKER_ARGS+=" --runtime nvidia"
else
    # x86_64 or others
    echo "🔵 Detected x86_64 platform"
    DOCKER_ARGS+=" --gpus all"
fi

docker run $DOCKER_ARGS \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e GDK_SCALE=2 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --network host \
     -v /dev:/dev \
    --device-cgroup-rule='c 81:* rmw' \
    --device-cgroup-rule='c 234:* rmw' \
    --shm-size=16gb \
    uniflexai/tinynav:a86a62a /tinynav/scripts/run_rosbag_examples.sh
