#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root/app/frontend"
flutter build web --release "$@"
python3 "$root/scripts/version_web_assets.py"
