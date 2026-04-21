#!/bin/bash
set -e

FLUTTER_INSTALL_DIR="$HOME/flutter"
FLUTTER_VERSION="3.32.0"
FLUTTER_TAR="flutter_linux_${FLUTTER_VERSION}-stable.tar.xz"
FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/${FLUTTER_TAR}"

TINYNAV_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="$TINYNAV_ROOT/app/frontend"
BACKEND_PORT=8000
FRONTEND_PORT=8080

# ── 1. Ensure Flutter ──────────────────────────────────────────────────────────
if command -v flutter &>/dev/null; then
    echo "[flutter] found at $(command -v flutter)"
elif [ -x "$FLUTTER_INSTALL_DIR/bin/flutter" ]; then
    echo "[flutter] found at $FLUTTER_INSTALL_DIR, adding to PATH"
    export PATH="$FLUTTER_INSTALL_DIR/bin:$PATH"
else
    echo "[flutter] not found — downloading Flutter $FLUTTER_VERSION..."
    curl -L "$FLUTTER_URL" -o "/tmp/$FLUTTER_TAR"
    tar -xf "/tmp/$FLUTTER_TAR" -C "$HOME"
    rm "/tmp/$FLUTTER_TAR"
    export PATH="$FLUTTER_INSTALL_DIR/bin:$PATH"
    echo "[flutter] installed at $FLUTTER_INSTALL_DIR"
fi

flutter --version

# ── 2. Build web ───────────────────────────────────────────────────────────────
echo "[frontend] building Flutter web..."
cd "$FRONTEND_DIR"
flutter pub get
flutter build web --release
echo "[frontend] build done → $FRONTEND_DIR/build/web"

# ── 3. Start backend ───────────────────────────────────────────────────────────
echo "[backend] starting on port $BACKEND_PORT..."
cd "$TINYNAV_ROOT"
source /opt/ros/humble/setup.bash
TINYNAV_DB_PATH="${TINYNAV_DB_PATH:-$TINYNAV_ROOT/tinynav_db}" \
    uv run uvicorn app.backend.main:app --host 0.0.0.0 --port "$BACKEND_PORT" &
BACKEND_PID=$!
echo "[backend] PID=$BACKEND_PID"

# ── 4. Start frontend static server ───────────────────────────────────────────
echo "[frontend] serving on port $FRONTEND_PORT..."
cd "$FRONTEND_DIR/build/web"
python3 -m http.server "$FRONTEND_PORT" &
FRONTEND_PID=$!
echo "[frontend] PID=$FRONTEND_PID"

echo ""
echo "  Backend:  http://0.0.0.0:$BACKEND_PORT"
echo "  Frontend: http://0.0.0.0:$FRONTEND_PORT"
echo ""
echo "Press Ctrl+C to stop both."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
