#!/bin/bash

FRONTEND_PORT="${FRONTEND_PORT:-80}"
BACKEND_MODE="${BACKEND_MODE:-display}"  # display | proxy | none
DISPLAY_BACKEND_PORT="${DISPLAY_BACKEND_PORT:-8000}"
TINYNAV_MANAGER_BASE_URL="${TINYNAV_MANAGER_BASE_URL:-http://169.254.10.1:8000}"
TINYNAV_DB_PATH="${TINYNAV_DB_PATH:-/tinynav/tinynav_db}"
DEVICE_PROXY_LISTEN_HOST="${DEVICE_PROXY_LISTEN_HOST:-0.0.0.0}"
DEVICE_PROXY_LISTEN_PORT="${DEVICE_PROXY_LISTEN_PORT:-8000}"
DEVICE_PROXY_TARGET_HOST="${DEVICE_PROXY_TARGET_HOST:-169.254.10.1}"
DEVICE_PROXY_TARGET_PORT="${DEVICE_PROXY_TARGET_PORT:-8000}"
DEVICE_PROXY_SCRIPT="/tmp/tinynav_device_proxy.py"

cat > "$DEVICE_PROXY_SCRIPT" <<'PY'
import socket
import threading

LHOST = "__LHOST__"
LPORT = __LPORT__
THOST = "__THOST__"
TPORT = __TPORT__


def pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except Exception:
            pass


def handle(client):
    try:
        target = socket.create_connection((THOST, TPORT), timeout=5)
    except Exception:
        client.close()
        return

    t1 = threading.Thread(target=pump, args=(client, target), daemon=True)
    t2 = threading.Thread(target=pump, args=(target, client), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    client.close()
    target.close()


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((LHOST, LPORT))
server.listen(64)
print(f"proxy {LHOST}:{LPORT} -> {THOST}:{TPORT}", flush=True)

while True:
    c, _ = server.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
PY

sed -i "s/__LHOST__/$DEVICE_PROXY_LISTEN_HOST/; s/__LPORT__/$DEVICE_PROXY_LISTEN_PORT/; s/__THOST__/$DEVICE_PROXY_TARGET_HOST/; s/__TPORT__/$DEVICE_PROXY_TARGET_PORT/" "$DEVICE_PROXY_SCRIPT"

# Stop any stale background proxy from older script versions.
pkill -f "$DEVICE_PROXY_SCRIPT" >/dev/null 2>&1 || true

tmux kill-session -t app >/dev/null 2>&1 || true

FRONTEND_CMD="python -m http.server $FRONTEND_PORT --directory /tinynav/app/frontend/build/web"
DISPLAY_BACKEND_CMD="source /opt/ros/humble/setup.bash 2>/dev/null || true; cd /tinynav && TINYNAV_BACKEND_ROLE=display TINYNAV_MANAGER_BASE_URL=$TINYNAV_MANAGER_BASE_URL TINYNAV_DB_PATH=$TINYNAV_DB_PATH python3 -m uvicorn app.backend.main:app --host 0.0.0.0 --port $DISPLAY_BACKEND_PORT"
PROXY_CMD="python3 $DEVICE_PROXY_SCRIPT"

if [ "$BACKEND_MODE" = "display" ]; then
  tmux new-session -s app \; \
    split-window -h \; \
    select-pane -t 0 \; send-keys "$FRONTEND_CMD" C-m \; \
    select-pane -t 1 \; send-keys "$DISPLAY_BACKEND_CMD" C-m
elif [ "$BACKEND_MODE" = "proxy" ]; then
  tmux new-session -s app \; \
    split-window -h \; \
    select-pane -t 0 \; send-keys "$FRONTEND_CMD" C-m \; \
    select-pane -t 1 \; send-keys "$PROXY_CMD" C-m
else
  tmux new-session -s app \; \
    send-keys "$FRONTEND_CMD" C-m
fi
