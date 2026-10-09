#!/bin/bash
# Go2 EKF beside the app (host side): build | start | stop | status | rm. Output /go2/ekf/odometry, not used by control.
# start creates the container once with --restart unless-stopped, so it comes back after every reboot until `stop`.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
IMG=tinynav-go2-ekf:local
case "$1" in
  build) docker build -t $IMG "$REPO/scripts/go2_ekf" ;;
  start)
    docker image inspect $IMG >/dev/null 2>&1 || docker build -t $IMG "$REPO/scripts/go2_ekf"
    if docker container inspect go2-ekf >/dev/null 2>&1; then docker start go2-ekf; else
      # host network + host /dev/shm: FastDDS shared memory with the app container (which mounts /dev)
      docker run -d --restart unless-stopped --name go2-ekf --net host -v /dev/shm:/dev/shm -e ROBOT_TYPE=go2 \
        -v "$REPO":/tinynav --entrypoint bash $IMG /tinynav/scripts/go2_ekf/entry.sh
    fi ;;
  stop) docker stop go2-ekf ;;                        # stays stopped across reboots
  rm) docker rm -f go2-ekf ;;                         # after changing the run options above; then start
  status) docker ps -a --filter name=go2-ekf --format '{{.Names}} {{.Status}}'; docker logs --tail 20 go2-ekf 2>&1 ;;
  *) echo "usage: $0 build|start|stop|status|rm"; exit 1 ;;
esac
