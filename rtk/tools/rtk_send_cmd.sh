#!/bin/bash
# Write one command to the UM982. The reply shows up in rtk/log/*_raw.log;
# never cat the pty for it, that makes a second reader.
set -u
CMD=${1:?usage: rtk_send_cmd.sh "COMMAND"}
PORT=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
python3 -c '
import os, sys
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_NOCTTY | os.O_NONBLOCK)
os.write(fd, (sys.argv[2] + "\r\n").encode())
os.close(fd)' "$PORT" "$CMD"
echo "$(date +%H:%M:%S) sent: $CMD"
