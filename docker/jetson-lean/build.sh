#!/bin/bash
# Build the Jetson lean image (docker/jetson-lean/Dockerfile) from this checkout.
#
#   On the Jetson itself (versions are read from the host):
#     bash docker/jetson-lean/build.sh [--tag TAG] [--no-app]
#   On another machine, through arm64 emulation (qemu-user-static / binfmt), naming the
#   TensorRT version of the Jetsons it is for (dpkg-query -W libnvinfer10 on one of them):
#     bash docker/jetson-lean/build.sh --cross --trt-version 10.3.0.30-1+cuda12.5 [--tag TAG] [--no-app]
#
#   --tag TAG       image tag (default tinynav:jetson-lean)
#   --no-app        leave out the web app and Flutter (smaller, faster to build)
#   --base IMAGE    an arm64 ros:humble to start from (default ros:humble)
#   anything after -- goes to docker build as is
#
# Run `git lfs pull` first: the image copies this checkout, models included.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
tag=tinynav:jetson-lean
cross=0
trt=""
args=()
while [ $# -gt 0 ]; do
  case $1 in
    --tag) tag=$2; shift 2 ;;
    --no-app) args+=(--build-arg WITH_APP=0); shift ;;
    --base) args+=(--build-arg BASE="$2"); shift 2 ;;
    --cross) cross=1; shift ;;
    --trt-version) trt=$2; shift 2 ;;
    --) shift; break ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$cross" = 1 ]; then
  if [[ ! "$trt" =~ ^10\.3\. ]]; then
    echo "--cross needs --trt-version 10.3.x (the exact libnvinfer10 version of the target Jetsons)" >&2
    exit 2
  fi
  args=(--platform linux/arm64 --build-arg TRT_VERSION="$trt" --build-arg CUDA_VERSION=12.6 "${args[@]}")
else
  if ! host_args=$(bash "$here/host_check.sh" --build-args); then
    echo "This host cannot build for itself (above). On another machine use --cross." >&2
    exit 1
  fi
  read -r -a host_args <<<"$(tr '\n' ' ' <<<"$host_args")"
  args=("${host_args[@]}" "${args[@]}")
fi

cd "$repo"
docker build "${args[@]}" "$@" -f docker/jetson-lean/Dockerfile -t "$tag" .
docker image inspect -f '{{.Id}} {{.Architecture}} {{.Size}}' "$tag"
