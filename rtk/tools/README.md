# RTK diagnostics (UM982, Beijing dog)

Host-side scripts, run by the `dm` crontab at boot. Output goes to `rtk/log/` (gitignored).

| Script | What it does |
|---|---|
| `rtk_diag_logger.sh` | Keeps `rtk_diag_logger.py` attached to the bridge's raw pty `/tmp/rtk_nmea` (inside `tinynav-dev`) and reconnects after container restarts. |
| `rtk_diag_logger.py` | Writes every receiver line to `rtk_YYYYMMDD_raw.log` and a 1 Hz summary to `rtk_YYYYMMDD.csv`. |
| `rtk_agnss_inject.py` | At boot, sends `$AIDPOS` (fixed site) and, once NTP is synced, `$AIDTIME`; confirms each from the receiver echo in the raw log. |
| `rtk_send_cmd.sh "CMD"` | Writes one command to the receiver; read the reply from the raw log. |

The pty must have a single reader. Do not `cat /tmp/rtk_nmea` while the logger runs: each line goes to only one reader, so both sides lose data.

Crontab (`crontab -e` as `dm`):

```
@reboot /home/dm/workspace/tinynav/rtk/tools/rtk_diag_logger.sh >/home/dm/workspace/tinynav/rtk/log/diag_logger.out 2>&1
@reboot sleep 15 && /usr/bin/python3 /home/dm/workspace/tinynav/rtk/tools/rtk_agnss_inject.py >>/home/dm/workspace/tinynav/rtk/log/agnss_inject.log 2>&1
```

A/B without AGNSS: `touch rtk/log/agnss_off` before rebooting, `rm` it afterwards.

Receiver logs these scripts rely on, saved to NVM with `SAVECONFIG`: `RTKSTATUSA 1`, `AGNSSSTATUSA 1`, `BESTSATA 1`, `RTCMSTATUSA ONCHANGED` (no `LOG` prefix; `LOG ... ONTIME` is rejected for these).
