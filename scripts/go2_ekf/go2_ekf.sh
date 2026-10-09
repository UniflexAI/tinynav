#!/bin/bash
# Go2 EKF beside the app (host side): build | start | stop | status. Output /go2/ekf/odometry, not used by control.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
IMG=tinynav-go2-ekf:local
case "$1" in
  build) docker build -t $IMG "$REPO/scripts/go2_ekf" ;;
  start)
    docker image inspect $IMG >/dev/null 2>&1 || docker build -t $IMG "$REPO/scripts/go2_ekf"
    # host network + host /dev/shm: FastDDS shared memory with the app container (which mounts /dev)
    docker run -d --rm --name go2-ekf --net host -v /dev/shm:/dev/shm -e ROBOT_TYPE=go2 \
      -v "$REPO":/tinynav --entrypoint bash $IMG /tinynav/scripts/go2_ekf/entry.sh ;;
  stop) docker stop go2-ekf ;;
  status) docker ps --filter name=go2-ekf --format '{{.Names}} {{.Status}}'; docker logs --tail 20 go2-ekf 2>&1 ;;
  *) echo "usage: $0 build|start|stop|status"; exit 1 ;;
esac
