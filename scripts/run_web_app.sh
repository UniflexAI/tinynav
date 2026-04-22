#!/bin/bash
set -e

# ── Colors & symbols ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}  ✔  ${RESET}$*"; }
info() { echo -e "${CYAN}  ➜  ${RESET}$*"; }
warn() { echo -e "${YELLOW}  ⚠  ${RESET}$*"; }
die()  { echo -e "${RED}  ✖  ${RESET}$*" >&2; exit 1; }
step() { echo -e "\n${BOLD}${CYAN}━━  $*  ━━${RESET}"; }

banner() {
  echo -e "${BOLD}${CYAN}"
  echo "  ████████╗██╗███╗   ██╗██╗   ██╗███╗   ██╗ █████╗ ██╗   ██╗"
  echo "     ██╔══╝██║████╗  ██║╚██╗ ██╔╝████╗  ██║██╔══██╗██║   ██║"
  echo "     ██║   ██║██╔██╗ ██║ ╚████╔╝ ██╔██╗ ██║███████║██║   ██║"
  echo "     ██║   ██║██║╚██╗██║  ╚██╔╝  ██║╚██╗██║██╔══██║╚██╗ ██╔╝"
  echo "     ██║   ██║██║ ╚████║   ██║   ██║ ╚████║██║  ██║ ╚████╔╝ "
  echo "     ╚═╝   ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝  ╚═══╝  Web App"
  echo -e "${RESET}"
}

spinner() {
  local pid=$1 msg=$2
  local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    printf "\r${CYAN}  ${frames[$((i % 10))]}  ${RESET}${DIM}%s${RESET}" "$msg"
    sleep 0.1
    ((i++))
  done
  printf "\r\033[K"
}

# ── Config ────────────────────────────────────────────────────────────────────
FLUTTER_INSTALL_DIR="$HOME/flutter"
FLUTTER_VERSION="3.32.0"
FLUTTER_TAR="flutter_linux_${FLUTTER_VERSION}-stable.tar.xz"
FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/${FLUTTER_TAR}"

TINYNAV_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="$TINYNAV_ROOT/app/frontend"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"

banner

# ── Port check ────────────────────────────────────────────────────────────────
for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if lsof -iTCP:"$port" -sTCP:LISTEN -t &>/dev/null 2>&1 || \
     ss -tlnp "sport = :$port" 2>/dev/null | grep -q ":$port"; then
    die "Port $port is already in use. Set BACKEND_PORT / FRONTEND_PORT to override."
  fi
done

# ── 1. Flutter ────────────────────────────────────────────────────────────────
step "Flutter"
if command -v flutter &>/dev/null; then
  ok "Flutter found: $(command -v flutter)"
elif [ -x "$FLUTTER_INSTALL_DIR/bin/flutter" ]; then
  export PATH="$FLUTTER_INSTALL_DIR/bin:$PATH"
  ok "Flutter found: $FLUTTER_INSTALL_DIR"
else
  warn "Flutter not found — downloading $FLUTTER_VERSION..."
  curl -L --progress-bar "$FLUTTER_URL" -o "/tmp/$FLUTTER_TAR"
  info "Extracting..."
  tar -xf "/tmp/$FLUTTER_TAR" -C "$HOME"
  rm "/tmp/$FLUTTER_TAR"
  export PATH="$FLUTTER_INSTALL_DIR/bin:$PATH"
  ok "Flutter installed at $FLUTTER_INSTALL_DIR"
fi
echo -e "  ${DIM}$(flutter --version 2>&1 | head -1)${RESET}"

# ── 2. Build web ──────────────────────────────────────────────────────────────
step "Build Flutter Web"
cd "$FRONTEND_DIR"
info "flutter pub get"
flutter pub get --suppress-analytics

info "flutter build web --release"
flutter build web --release --suppress-analytics 2>&1 | \
  grep -v "^$" | sed "s/^/  ${DIM}/" | sed "s/$/${RESET}/" || true
ok "Build complete → app/frontend/build/web"

# ── 3. Backend ────────────────────────────────────────────────────────────────
step "Start Backend"
cd "$TINYNAV_ROOT"
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
export TINYNAV_DB_PATH="${TINYNAV_DB_PATH:-$TINYNAV_ROOT/tinynav_db}"
info "TINYNAV_DB_PATH=$TINYNAV_DB_PATH"

export UNITREE_NETWORK_INTERFACE="${UNITREE_NETWORK_INTERFACE:-enP8p1s0}"
info "UNITREE_NETWORK_INTERFACE=$UNITREE_NETWORK_INTERFACE"

uv run uvicorn app.backend.main:app \
  --host 0.0.0.0 --port "$BACKEND_PORT" \
  --log-level warning &
BACKEND_PID=$!

# Wait for backend to be ready
for i in $(seq 1 20); do
  sleep 0.5
  if curl -sf "http://127.0.0.1:$BACKEND_PORT/device/info" &>/dev/null; then
    ok "Backend ready  (PID $BACKEND_PID)"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    die "Backend failed to start."
  fi
  [ "$i" -eq 20 ] && warn "Backend may still be starting up..."
done

# ── 4. Frontend ───────────────────────────────────────────────────────────────
# Flutter CanvasKit (WASM) requires Cross-Origin-Opener-Policy + Cross-Origin-Embedder-Policy
# headers for SharedArrayBuffer support — python3 -m http.server doesn't set them,
# which causes a white screen on mobile browsers.
step "Start Frontend"
_PYSERVER=$(mktemp /tmp/tinynav_server_XXXX.py)
cat > "$_PYSERVER" << 'PYEOF'
import http.server, sys, os
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()
    def log_message(self, *a): pass
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
with http.server.HTTPServer(('0.0.0.0', port), H) as s:
    s.serve_forever()
PYEOF
cd "$FRONTEND_DIR/build/web"
python3 "$_PYSERVER" "$FRONTEND_PORT" &>/dev/null &
FRONTEND_PID=$!
sleep 0.3
kill -0 "$FRONTEND_PID" 2>/dev/null || die "Frontend server failed to start."
ok "Frontend ready  (PID $FRONTEND_PID)"

# ── Ready ─────────────────────────────────────────────────────────────────────
LOCAL_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
echo -e "\n${BOLD}${GREEN}  ✦  TinyNav App is running${RESET}\n"
echo -e "  ${BOLD}Frontend${RESET}  http://${LOCAL_IP:-localhost}:${FRONTEND_PORT}"
echo -e "  ${BOLD}Backend ${RESET}  http://${LOCAL_IP:-localhost}:${BACKEND_PORT}/docs"
echo -e "\n  ${DIM}Press Ctrl+C to stop.${RESET}\n"

trap "echo -e '\n${YELLOW}  Stopping...${RESET}'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; rm -f '$_PYSERVER'; ok 'Stopped.'; exit 0" INT TERM
wait
