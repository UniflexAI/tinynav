#!/usr/bin/env bash
set -euo pipefail

mkdir -p /userdata/tmp
export TMPDIR="${TMPDIR:-/userdata/tmp}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"

python3 -m pip install --upgrade pip

# Wheels may be unavailable on this device arch; fall back to source installs.
python3 -m pip install decord || python3 -m pip install 'git+https://github.com/dmlc/decord'
python3 -m pip install pydbow3 || python3 -m pip install 'git+https://github.com/JHMeusener/PyDBoW3.git'

python3 - <<'PY'
import decord
import pydbow3
print('decord_ok', getattr(decord, '__version__', 'unknown'))
print('pydbow3_ok', getattr(pydbow3, '__name__', 'pydbow3'))
PY
