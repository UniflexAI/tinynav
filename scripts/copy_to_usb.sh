#!/usr/bin/env bash
#
# Copy a folder from tinynav_db to a USB drive, then unmount.
#
# Usage:
#   ./scripts/copy_to_usb.sh <folder_name> [folder_name ...]
#
# Example:
#   ./scripts/copy_to_usb.sh maps/map_back
#   ./scripts/copy_to_usb.sh rosbags/bag_2026_07_15_17_55_36
#
# The folder path is relative to TINYNAV_DB_PATH (default: /tinynav/tinynav_db).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB_PATH="${TINYNAV_DB_PATH:-/tinynav/tinynav_db}"
USB_DEV="${USB_DEV:-/dev/sda}"
MOUNT_POINT="${USB_MOUNT_POINT:-/media/dm/DM_DATA}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <folder_name> [folder_name ...]"
  echo "  folder_name is relative to $DB_PATH"
  echo ""
  echo "Available folders:"
  ls -1 "$DB_PATH" 2>/dev/null || echo "  (DB path not found: $DB_PATH)"
  exit 1
fi

# Mount USB if not already mounted
if ! mountpoint -q "$MOUNT_POINT"; then
  echo "Mounting $USB_DEV at $MOUNT_POINT ..."
  sudo mount "$USB_DEV" "$MOUNT_POINT" 2>&1 || {
    echo "ERROR: Failed to mount $USB_DEV" >&2
    exit 1
  }
fi

mkdir -p "$MOUNT_POINT"

total=$#
idx=0
failed=()

for rel_path in "$@"; do
  idx=$((idx + 1))
  SRC="$DB_PATH/$rel_path"
  DEST_DIR="$MOUNT_POINT/$rel_path"

  if [[ ! -d "$SRC" ]]; then
    echo "[$idx/$total] Source not found: $SRC" >&2
    failed+=("$rel_path")
    continue
  fi

  # Skip if already copied (compare a representative file size)
  if [[ -d "$DEST_DIR" ]]; then
    src_size=$(du -sb "$SRC" 2>/dev/null | cut -f1)
    dest_size=$(du -sb "$DEST_DIR" 2>/dev/null | cut -f1)
    if [[ -n "$src_size" && -n "$dest_size" && "$src_size" == "$dest_size" ]]; then
      echo "[$idx/$total] Skip $rel_path (already complete, ${dest_size} bytes)"
      continue
    fi
    echo "[$idx/$total] Partial copy detected, re-copying ..."
    rm -rf "$DEST_DIR"
  fi

  echo "[$idx/$total] Copying $rel_path ..."
  echo "  from: $SRC"
  echo "  to:   $DEST_DIR"
  mkdir -p "$(dirname "$DEST_DIR")"
  if cp -a "$SRC" "$DEST_DIR"; then
    sync
    echo "[$idx/$total] Done: $rel_path"
    df -h "$MOUNT_POINT" | tail -1
  else
    echo "[$idx/$total] FAILED: $rel_path" >&2
    failed+=("$rel_path")
    rm -rf "$DEST_DIR"
  fi
  echo
done

sync

# Unmount USB
echo "Unmounting $MOUNT_POINT ..."
sudo umount "$MOUNT_POINT" 2>&1 || {
  echo "WARNING: Failed to unmount $MOUNT_POINT (busy?)" >&2
}

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Finished with errors. Failed (${#failed[@]}): ${failed[*]}" >&2
  exit 1
fi

echo "All $total folder(s) copied to USB and unmounted successfully."
