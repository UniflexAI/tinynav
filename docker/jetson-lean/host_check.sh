#!/bin/bash
# Is this Jetson able to run the lean image, which takes CUDA, cuDNN and TensorRT from
# the host's JetPack instead of carrying them?
#
#   bash docker/jetson-lean/host_check.sh               # the verdict, one line per check
#   bash docker/jetson-lean/host_check.sh --build-args  # docker build arguments for this host
#   bash docker/jetson-lean/host_check.sh --mounts      # docker run -v arguments, one per line
#
# Supported: Jetson Orin, JetPack 6.1 - 6.2.x (L4T R36.4 or later in R36): CUDA 12.6,
# cuDNN 9, TensorRT 10.3. Nothing on the host is changed.
#
# TINYNAV_HOST_ROOT (default empty) prefixes every host path, and DPKG_QUERY names the
# dpkg-query to use; both exist for testing this script off a Jetson.
set -euo pipefail

ROOT="${TINYNAV_HOST_ROOT:-}"
DPKG_QUERY="${DPKG_QUERY:-dpkg-query}"
LIB=/usr/lib/aarch64-linux-gnu
CUDA_VERSION=12.6
TRT_SERIES=10.3
# The libraries the image takes from the host: every TensorRT (libnvinfer*, including
# the builder resource trtexec loads), ONNX parser and cuDNN library (cuDNN 9 is split
# into libcudnn_ops, libcudnn_cnn, libcudnn_engines_* ...), with their version links.
PATTERNS=(libnvinfer libnvonnxparser libcudnn)
REQUIRED=(libnvinfer.so.10 libnvinfer_plugin.so.10 libnvonnxparser.so.10 libcudnn.so.9)

mode=check
case "${1:-}" in
  "") ;;
  --build-args) mode=build-args ;;
  --mounts) mode=mounts ;;
  --no-docker) mode=check-no-docker ;;
  -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

problems=()
note() { [ "$mode" = check ] || [ "$mode" = check-no-docker ] && echo "$@" || true; }
fail() { problems+=("$1"); note "❌ $1"; }

# 1. A Jetson: aarch64 with an L4T release file.
arch="${TINYNAV_ARCH:-$(uname -m)}"
if [ "$arch" != aarch64 ]; then
  fail "this is $arch, not a Jetson (aarch64): use the root Dockerfile on other machines"
fi
release_file="$ROOT/etc/nv_tegra_release"
if [ ! -f "$release_file" ]; then
  fail "no $release_file: not a Jetson running L4T/JetPack"
else
  # e.g. "# R36 (release), REVISION: 4.3, GCID: ..., BOARD: generic, EABI: aarch64, ..."
  line=$(head -n 1 "$release_file")
  major=$(sed -n 's/^# R\([0-9]*\) .*/\1/p' <<<"$line")
  revision=$(sed -n 's/.*REVISION: \([0-9.]*\).*/\1/p' <<<"$line")
  minor=${revision%%.*}
  if [ "$major" != 36 ] || [ -z "$minor" ] || [ "$minor" -lt 4 ]; then
    fail "L4T R${major:-?}.${revision:-?}: needs R36.4 or later in R36 (JetPack 6.1 - 6.2.x)"
  else
    note "✅ L4T R$major.$revision"
  fi
fi

# 2. TensorRT 10.3 (the binding baked into the image must match it exactly).
trt=$("$DPKG_QUERY" -W -f='${Version}' libnvinfer10 2>/dev/null || true)
if [ -z "$trt" ]; then
  fail "TensorRT (package libnvinfer10) is not installed: install the JetPack runtime (sudo apt install nvidia-jetpack)"
elif [ "${trt#"$TRT_SERIES".}" = "$trt" ]; then
  fail "TensorRT $trt: needs $TRT_SERIES.x"
else
  note "✅ TensorRT $trt"
fi

# 3. CUDA and the libraries themselves.
cuda_dir="$ROOT/usr/local/cuda-$CUDA_VERSION"
if [ -d "$cuda_dir" ]; then note "✅ CUDA $CUDA_VERSION"; else fail "no /usr/local/cuda-$CUDA_VERSION (CUDA $CUDA_VERSION)"; fi
for name in "${REQUIRED[@]}"; do
  if [ -e "$ROOT$LIB/$name" ]; then note "✅ $name"; else fail "no $LIB/$name"; fi
done

# 4. Docker with the NVIDIA runtime.
if [ "$mode" = check ]; then
  if ! command -v docker >/dev/null; then
    fail "Docker is not installed"
  elif ! docker info >/dev/null 2>&1; then
    fail "Docker is not running or this user may not use it"
  elif ! docker info 2>/dev/null | grep -i 'Runtimes' | grep -q nvidia; then
    fail "Docker has no nvidia runtime: sudo apt install nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  else
    note "✅ Docker with the nvidia runtime"
  fi
fi

if [ "${#problems[@]}" -gt 0 ]; then
  if [ "$mode" = build-args ] || [ "$mode" = mounts ]; then
    printf '%s\n' "${problems[@]}" | sed 's/^/host_check: /' >&2
  else
    echo "This host cannot run the Jetson lean image; see docker/jetson-lean/README.md."
  fi
  exit 1
fi

case "$mode" in
  build-args)
    echo "--build-arg TRT_VERSION=$trt"
    echo "--build-arg CUDA_VERSION=$CUDA_VERSION"
    ;;
  mounts)
    # Read-only, file by file: the container keeps its own /usr/lib/aarch64-linux-gnu.
    echo "/usr/local/cuda-$CUDA_VERSION:/usr/local/cuda-$CUDA_VERSION:ro"
    for base in "${PATTERNS[@]}"; do
      for f in "$ROOT$LIB/$base"*.so*; do
        [ -e "$f" ] || continue
        [[ "$(basename "$f")" =~ ^lib(nvinfer|nvonnxparser|cudnn)[^/]*\.so(\.[0-9.]+)?$ ]] || continue
        f=${f#"$ROOT"}
        echo "$f:$f:ro"
      done
    done
    ;;
  *)
    echo "This host can run the Jetson lean image."
    ;;
esac
