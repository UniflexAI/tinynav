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

# ── Config ────────────────────────────────────────────────────────────────────
TINYNAV_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="$TINYNAV_ROOT/app/frontend"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"

banner

# ── Help ──────────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --help|-h)
      echo "Usage: $0"
      echo "  Serves the pre-built frontend from app/frontend/build/web."
      echo "  Build it first with: cd app/frontend && flutter build web --release"
      exit 0 ;;
  esac
done

# ── Pre-check ─────────────────────────────────────────────────────────────────
if [ ! -f "$FRONTEND_DIR/build/web/index.html" ]; then
  die "No frontend build found at app/frontend/build/web — run 'flutter build web --release' first."
fi

for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if lsof -iTCP:"$port" -sTCP:LISTEN -t &>/dev/null 2>&1 || \
     ss -tlnp "sport = :$port" 2>/dev/null | grep -q ":$port"; then
    die "Port $port is already in use. Set BACKEND_PORT / FRONTEND_PORT to override."
  fi
done

# ── 1. Backend ────────────────────────────────────────────────────────────────
step "Start Backend"
cd "$TINYNAV_ROOT"
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
export TINYNAV_DB_PATH="${TINYNAV_DB_PATH:-$TINYNAV_ROOT/tinynav_db}"
info "TINYNAV_DB_PATH=$TINYNAV_DB_PATH"

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

# ── 2. Frontend ───────────────────────────────────────────────────────────────
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
