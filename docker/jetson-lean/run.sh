#!/bin/bash
# Start the Jetson lean image with the host's CUDA, cuDNN and TensorRT mounted read-only.
#
#   bash docker/jetson-lean/run.sh [--image IMAGE] [--name NAME] [--no-workspace] [-- COMMAND ...]
#
#   --image IMAGE    the image to run (default tinynav:jetson-lean, as build.sh tags it)
#   --name NAME      a container name
#   --no-workspace   use the code baked into the image instead of mounting this checkout
#                    at /tinynav (mounting it is what the dev container does)
#   COMMAND          what to run (default: bash)
#
# The other run arguments are .devcontainer/devcontainer.json's, with the NVIDIA runtime
# Jetsons use instead of --gpus.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
image=tinynav:jetson-lean
name=()
workspace=(-v "$repo:/tinynav")
while [ $# -gt 0 ]; do
  case $1 in
    --image) image=$2; shift 2 ;;
    --name) name=(--name "$2"); shift 2 ;;
    --no-workspace) workspace=(); shift ;;
    --) shift; break ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (put the command after --)" >&2; exit 2 ;;
  esac
done
command=("$@")
[ ${#command[@]} -gt 0 ] || command=(bash)

if ! verdict=$(bash "$here/host_check.sh" 2>&1); then
  echo "$verdict" >&2
  exit 1
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "No image $image: build it first (bash docker/jetson-lean/build.sh)." >&2
  exit 1
fi

gpu=()
while IFS= read -r mount; do
  gpu+=(-v "$mount")
done < <(bash "$here/host_check.sh" --mounts)

# The driver directory: newer NVIDIA container runtimes fill it themselves (and their
# hook writes a libcuda.so link into it, which a read-only mount would break); older
# ones leave it empty. Ask the runtime which it is.
driver=/usr/lib/aarch64-linux-gnu/nvidia
if ! docker run --rm --pull=never --runtime nvidia --network none --entrypoint ls "$image" -1 "$driver" 2>/dev/null \
    | grep -qx 'libcuda.so.1'; then
  gpu+=(-v "$driver:$driver:ro")
fi

tty=(-i)
[ -t 0 ] && [ -t 1 ] && tty=(-it)
mkdir -p "$HOME/.local/share/tinynav"
exec docker run "${tty[@]}" --rm "${name[@]}" \
  --runtime nvidia --privileged --network host \
  -e DISPLAY="${DISPLAY:-}" -e GDK_SCALE=2 \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev -v /etc/localtime:/etc/localtime \
  --device-cgroup-rule='c 81:* rwm' --device-cgroup-rule='c 234:* rwm' --shm-size=16gb \
  -v "$HOME/.local/share/tinynav:/root/.local/share/tinynav" \
  "${workspace[@]}" "${gpu[@]}" \
  "$image" "${command[@]}"
