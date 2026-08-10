#!/usr/bin/env python3
import base64
import calendar
import fcntl
import json
import math
import os
import pty
import select
import socket
import termios
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, QuaternionStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus, TimeReference
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
KNOT_TO_MPS = 0.5144444444444445

GGA_QUALITY_NAMES = {
    0: "NO_FIX",
    1: "SINGLE",
    2: "DGNSS",
    3: "GPS_PPS",
    4: "RTK_FIXED",
    5: "RTK_FLOAT",
    6: "INS",
    7: "MANUAL",
    8: "SIMULATOR",
}

NAVSAT_STATUS_NAMES = {
    int(NavSatStatus.STATUS_NO_FIX): "STATUS_NO_FIX",
    int(NavSatStatus.STATUS_FIX): "STATUS_FIX",
    int(NavSatStatus.STATUS_SBAS_FIX): "STATUS_SBAS_FIX",
    int(NavSatStatus.STATUS_GBAS_FIX): "STATUS_GBAS_FIX",
}

UNICORE_POSITION_TYPE_STAGES = {
    "NONE": "NO_FIX",
    "SINGLE": "SINGLE",
    "PSRDIFF": "DGNSS",
    "SBAS": "DGNSS",
    "L1_FLOAT": "RTK_FLOAT",
    "IONOFREE_FLOAT": "RTK_FLOAT",
    "NARROW_FLOAT": "RTK_FLOAT",
    "L1_INT": "RTK_FIXED",
    "WIDE_INT": "RTK_FIXED",
    "NARROW_INT": "RTK_FIXED",
    "INS": "INS",
    "INS_PSRSP": "INS",
    "INS_PSRDIFF": "DGNSS",
    "INS_RTKFLOAT": "RTK_FLOAT",
    "INS_RTKFIXED": "RTK_FIXED",
    "PPP_CONVERGING": "PPP",
}

RTK_CALCULATE_STATUS_NAMES = {
    0: "NO_DIFFERENTIAL_SOURCE",
    1: "SOURCE_INSUFFICIENT_OBS",
    2: "SOURCE_DELAY_TOO_LARGE",
    3: "IONOSPHERE_ACTIVE",
    4: "ROVER_INSUFFICIENT_OBS",
    5: "RTK_READY",
}


@dataclass
class LlaOrigin:
    lat_rad: float
    lon_rad: float
    alt_m: float
    ecef: np.ndarray
    ecef_to_enu: np.ndarray


def lla_to_ecef(lat_rad: float, lon_rad: float, alt_m: float) -> np.ndarray:
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (n + alt_m) * cos_lat * cos_lon,
            (n + alt_m) * cos_lat * sin_lon,
            (n * (1.0 - WGS84_E2) + alt_m) * sin_lat,
        ],
        dtype=np.float64,
    )


def make_origin(lat_deg: float, lon_deg: float, alt_m: float) -> LlaOrigin:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    ecef_to_enu = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return LlaOrigin(lat, lon, alt_m, lla_to_ecef(lat, lon, alt_m), ecef_to_enu)


def lla_to_enu(lat_deg: float, lon_deg: float, alt_m: float, origin: LlaOrigin) -> np.ndarray:
    ecef = lla_to_ecef(math.radians(lat_deg), math.radians(lon_deg), alt_m)
    return origin.ecef_to_enu @ (ecef - origin.ecef)


def yaw_to_quat(yaw_rad: float):
    return R.from_euler("z", yaw_rad).as_quat()


def nmea_checksum_ok(line: str) -> bool:
    if not line.startswith("$") or "*" not in line:
        return False
    body, checksum = line[1:].split("*", 1)
    value = 0
    for ch in body:
        value ^= ord(ch)
    try:
        expected = int(checksum[:2], 16)
    except ValueError:
        return False
    return value == expected


def parse_float_or_none(value: str):
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int_or_none(value: str):
    if value == "":
        return None
    try:
        return int(value, 0)
    except ValueError:
        return None


def gga_quality_name(quality: int) -> str:
    return GGA_QUALITY_NAMES.get(int(quality), f"QUALITY_{int(quality)}")


def navsat_status_name(status: int | None) -> str | None:
    if status is None:
        return None
    return NAVSAT_STATUS_NAMES.get(int(status), f"STATUS_{int(status)}")


def stage_from_gga_quality(quality: int) -> str:
    return {
        0: "NO_FIX",
        1: "SINGLE",
        2: "DGNSS",
        4: "RTK_FIXED",
        5: "RTK_FLOAT",
        6: "INS",
    }.get(int(quality), "UNKNOWN")


def position_type_from_gga_quality(quality: int) -> str | None:
    return {
        0: "NONE",
        1: "SINGLE",
        2: "PSRDIFF",
        4: "NARROW_INT",
        5: "NARROW_FLOAT",
        6: "INS",
    }.get(int(quality))


def stage_from_position_type(position_type: str | None) -> str | None:
    if not position_type:
        return None
    return UNICORE_POSITION_TYPE_STAGES.get(position_type.upper(), "UNKNOWN")


def nmea_latlon(value: str, hemi: str) -> float | None:
    if not value or not hemi:
        return None
    dot = value.find(".")
    deg_digits = (dot - 2) if dot >= 0 else (len(value) - 2)
    if deg_digits <= 0:
        return None
    deg = float(value[:deg_digits])
    minutes = float(value[deg_digits:])
    out = deg + minutes / 60.0
    if hemi in ("S", "W"):
        out = -out
    return out


def ros_time_from_utc(hhmmss: str, ddmmyy: str | None = None):
    if not hhmmss:
        return None
    now = datetime.now(timezone.utc)
    day = now.day
    month = now.month
    year = now.year
    if ddmmyy and len(ddmmyy) >= 6:
        day = int(ddmmyy[0:2])
        month = int(ddmmyy[2:4])
        year = 2000 + int(ddmmyy[4:6])
    hour = int(hhmmss[0:2])
    minute = int(hhmmss[2:4])
    sec_float = float(hhmmss[4:])
    second = int(sec_float)
    nanosec = int(round((sec_float - second) * 1e9))
    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    stamp = Time()
    stamp.sec = calendar.timegm(dt.utctimetuple())
    stamp.nanosec = nanosec
    return stamp


def set_serial_raw(fd: int, baud: int):
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = attrs[2] | termios.CLOCAL | termios.CREAD
    attrs[2] = attrs[2] & ~termios.CSIZE
    attrs[2] = attrs[2] | termios.CS8
    attrs[2] = attrs[2] & ~(termios.PARENB | termios.CSTOPB | termios.CRTSCTS)
    attrs[3] = 0
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 1
    speed = getattr(termios, f"B{baud}", None)
    if speed is None:
        raise ValueError(f"Unsupported baud rate: {baud}")
    attrs[4] = speed
    attrs[5] = speed
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


class RtkBridgeNode(Node):
    def __init__(self):
        super().__init__("rtk_bridge_node")
        self._declare_params()
        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.min_navsat_status = int(self.get_parameter("min_navsat_status").value)
        self.origin_min_navsat_status = int(self.get_parameter("origin_min_navsat_status").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.path_max_size = int(self.get_parameter("path_max_size").value)
        self.heading_offset = math.radians(float(self.get_parameter("heading_offset_deg").value))
        self.use_heading = bool(self.get_parameter("use_heading").value)
        self.heading_is_north_clockwise = bool(self.get_parameter("heading_is_north_clockwise").value)
        self.serial_enabled = bool(self.get_parameter("serial_enabled").value)
        self.serial_port = self.get_parameter("serial_port").value
        self.baud = int(self.get_parameter("baud").value)
        self.rtcm_serial_port = self.get_parameter("rtcm_serial_port").value or self.serial_port
        self.rtcm_baud = int(self.get_parameter("rtcm_baud").value) or self.baud
        self.ntrip_enabled = bool(self.get_parameter("ntrip_enabled").value)
        self.raw_pty_enabled = bool(self.get_parameter("raw_pty_enabled").value)
        self.raw_pty_path = self.get_parameter("raw_pty_path").value
        self.raw_sentence_types = self._parse_sentence_type_filter(self.get_parameter("raw_sentence_types").value)

        self.origin = self._load_origin_from_params()
        self.latest_fix: NavSatFix | None = None
        self.latest_gga = ""
        self.latest_position_gga = ""
        self.latest_gga_quality = 0
        self.latest_num_satellites = 0
        self.latest_hdop = float("nan")
        self.latest_gga_utc = ""
        self.latest_gga_differential_age = None
        self.latest_gga_station_id = ""
        self.latest_receiver_stage = "NO_FIX"
        self.latest_receiver_position_type = None
        self.latest_bestnav_solution_status = None
        self.latest_bestnav_position_type = None
        self.latest_bestnav_std = None
        self.latest_bestnav_diff_age = None
        self.latest_bestnav_station_id = ""
        self.latest_rtk_calculate_status = None
        self.latest_rtk_calculate_status_name = None
        self.latest_rtk_position_type = None
        self.latest_rtk_ion_detected = None
        self.latest_rtk_dual_rtk_flag = None
        self.latest_rtcm_status = None
        self.latest_uniloglist = None
        self.latest_unicore_log = ""
        self.latest_unicore_log_type = ""
        self.last_unicore_log_time = None
        self.unicore_log_count = 0
        self.unicore_status_count = 0
        self.latest_sentence = ""
        self.latest_sentence_type = ""
        self.last_gga_time = None
        self.last_fix_time = None
        self.nmea_checksum_fail_count = 0
        self.nmea_parse_error_count = 0
        self.nmea_sentence_count = 0
        self.nmea_gga_count = 0
        self.nmea_rmc_count = 0
        self.nmea_heading_count = 0
        self.raw_sentence_publish_count = 0
        self.status_seq = 0
        self.latest_heading_yaw = 0.0
        self.latest_heading_stamp = None
        self.latest_velocity = np.zeros(3, dtype=np.float64)
        self.latest_velocity_stamp = None
        self.latest_time_reference = None
        self.last_nmea_time = None
        self.ntrip_connected = False
        self.ntrip_connect_count = 0
        self.ntrip_disconnect_count = 0
        self.last_rtcm_time = None
        self.rtcm_bytes = 0
        self.rtcm_written_bytes = 0
        self.rtcm_dropped_bytes = 0
        self.rtcm_write_fail_count = 0
        self.raw_pty_dropped_lines = 0
        self.latest_ntrip_gga_source = "none"
        self.latest_enu = None
        self.serial_fd = None
        self.nmea_fd = None
        self.rtcm_fd = None
        self.raw_pty_master = None
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self.path = Path()
        self.path.header.frame_id = self.frame_id

        self.fix_pub = self.create_publisher(NavSatFix, self.get_parameter("fix_topic").value, 20)
        self.heading_pub = self.create_publisher(QuaternionStamped, self.get_parameter("heading_topic").value, 20)
        self.vel_pub = self.create_publisher(TwistStamped, self.get_parameter("vel_topic").value, 20)
        self.time_ref_pub = self.create_publisher(TimeReference, self.get_parameter("time_reference_topic").value, 20)
        self.odom_pub = self.create_publisher(Odometry, self.get_parameter("odom_topic").value, 10)
        self.path_pub = self.create_publisher(Path, self.get_parameter("path_topic").value, 10)
        self.status_pub = self.create_publisher(String, self.get_parameter("status_topic").value, 10)
        self.io_status_pub = self.create_publisher(String, self.get_parameter("io_status_topic").value, 10)
        self.receiver_status_pub = self.create_publisher(
            String, self.get_parameter("receiver_status_topic").value, 10
        )
        self.raw_pub = self.create_publisher(String, self.get_parameter("raw_sentence_topic").value, 50)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.status_timer = self.create_timer(1.0, self._publish_status_timer)

        if self.raw_pty_enabled:
            self._setup_raw_pty()
        if self.serial_enabled:
            self._open_serial()
            threading.Thread(target=self._serial_loop, daemon=True).start()
        if self.ntrip_enabled:
            threading.Thread(target=self._ntrip_loop, daemon=True).start()

        self.get_logger().info(
            f"RTK bridge ready. serial={self.serial_enabled}:{self.serial_port}, "
            f"ntrip={self.ntrip_enabled}, odom={self.get_parameter('odom_topic').value}"
        )

    def _declare_params(self):
        self.declare_parameter("fix_topic", "/fix")
        self.declare_parameter("heading_topic", "/heading")
        self.declare_parameter("vel_topic", "/vel")
        self.declare_parameter("time_reference_topic", "/time_reference")
        self.declare_parameter("odom_topic", "/rtk/odom")
        self.declare_parameter("path_topic", "/rtk/path")
        self.declare_parameter("status_topic", "/rtk/status")
        self.declare_parameter("io_status_topic", "/rtk/io_status")
        self.declare_parameter("receiver_status_topic", "/rtk/receiver_status")
        self.declare_parameter("raw_sentence_topic", "/rtk/nmea_sentence")
        self.declare_parameter("frame_id", "rtk_world")
        self.declare_parameter("child_frame_id", "rtk_base")
        self.declare_parameter("min_navsat_status", int(NavSatStatus.STATUS_FIX))
        # ENU origin is only locked once a fix reaches this status, so a poor
        # early single-point fix cannot bias the whole session's local frame
        # (most visibly the altitude). Default: RTK float/fixed.
        self.declare_parameter("origin_min_navsat_status", int(NavSatStatus.STATUS_GBAS_FIX))
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("path_max_size", 2000)
        self.declare_parameter("heading_offset_deg", 0.0)
        # Default off: single-antenna setups have no usable heading (THS is then
        # course-over-ground noise). When false the bridge neither publishes
        # /heading nor feeds yaw into /rtk/odom. Enable only with a verified
        # dual-antenna heading solution.
        self.declare_parameter("use_heading", False)
        self.declare_parameter("heading_is_north_clockwise", True)
        self.declare_parameter("origin_lat", float("nan"))
        self.declare_parameter("origin_lon", float("nan"))
        self.declare_parameter("origin_alt", float("nan"))
        self.declare_parameter("serial_enabled", True)
        self.declare_parameter("serial_port", "/dev/ttyCH341USB0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("rtcm_serial_port", "")
        self.declare_parameter("rtcm_baud", 0)
        self.declare_parameter("split_same_serial_fd", True)
        self.declare_parameter("serial_read_only", True)
        self.declare_parameter("serial_init_commands", "")
        self.declare_parameter("rtcm_serial_init_commands", "")
        # 0 disables. Well above the receiver's 10 Hz output, far below the
        # minutes of silence a stalled USB read endpoint costs.
        self.declare_parameter("nmea_stale_restart_s", 15.0)
        self.declare_parameter("raw_pty_enabled", True)
        self.declare_parameter("raw_pty_path", "/tmp/rtk_nmea")
        self.declare_parameter("raw_sentence_types", "GGA")
        self.declare_parameter("ntrip_enabled", True)
        self.declare_parameter("ntrip_host", os.environ.get("TINYNAV_NTRIP_HOST", "120.253.239.161"))
        self.declare_parameter("ntrip_port", int(os.environ.get("TINYNAV_NTRIP_PORT", "8002")))
        self.declare_parameter("ntrip_mountpoint", os.environ.get("TINYNAV_NTRIP_MOUNTPOINT", "RTCM33_GRCEJ"))
        self.declare_parameter("ntrip_user", os.environ.get("TINYNAV_NTRIP_USER", ""))
        self.declare_parameter("ntrip_password", os.environ.get("TINYNAV_NTRIP_PASSWORD", ""))
        self.declare_parameter("ntrip_request_version", os.environ.get("TINYNAV_NTRIP_REQUEST_VERSION", "1.0"))
        self.declare_parameter("ntrip_initial_gga", os.environ.get("TINYNAV_NTRIP_INITIAL_GGA", "$GNGGA,085520.00,2246.89808758,N,11330.83046100,E,1,17,1.1,5.5866,M,-5.5511,M,,*6F"))
        self.declare_parameter("ntrip_gga_period_s", 1.0)
        self.declare_parameter("ntrip_reconnect_s", 3.0)
        self.declare_parameter("ntrip_recv_timeout_s", 1.0)
        self.declare_parameter("rtcm_write_timeout_s", 0.25)
        self.declare_parameter("fix_stale_after_s", 2.0)
        self.declare_parameter("strict_nmea_checksum", False)

    def _setup_raw_pty(self):
        master, slave = pty.openpty()
        slave_name = os.ttyname(slave)
        flags = fcntl.fcntl(master, fcntl.F_GETFL)
        fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            if os.path.islink(self.raw_pty_path) or os.path.exists(self.raw_pty_path):
                os.unlink(self.raw_pty_path)
            os.symlink(slave_name, self.raw_pty_path)
        except OSError as exc:
            self.get_logger().warning(f"Could not create {self.raw_pty_path}: {exc}")
        self.raw_pty_master = master
        self.get_logger().info(f"Raw NMEA mirror: cat {self.raw_pty_path}")

    def _mirror_raw_sentence(self, line: str):
        if self.raw_pty_master is None:
            return
        try:
            os.write(self.raw_pty_master, (line + "\n").encode("ascii", errors="ignore"))
        except BlockingIOError:
            self.raw_pty_dropped_lines += 1
        except OSError:
            pass

    def _open_serial(self):
        if self.rtcm_serial_port == self.serial_port and self.rtcm_baud != self.baud:
            raise ValueError("rtcm_baud cannot differ from baud when both streams use the same serial port")
        split_same_port = self.rtcm_serial_port == self.serial_port and bool(
            self.get_parameter("split_same_serial_fd").value
        )
        read_only = bool(self.get_parameter("serial_read_only").value)
        nmea_flags = os.O_RDWR if self.rtcm_serial_port == self.serial_port and not split_same_port else (
            os.O_RDONLY if read_only else os.O_RDWR
        )
        self.nmea_fd = os.open(self.serial_port, nmea_flags | os.O_NOCTTY | os.O_NONBLOCK)
        set_serial_raw(self.nmea_fd, self.baud)
        self.serial_fd = self.nmea_fd
        self.get_logger().info(f"Opened RTK serial {self.serial_port} at {self.baud}")
        self._send_serial_init_commands(self.nmea_fd, "serial_init_commands", self.serial_port)
        if self.rtcm_serial_port == self.serial_port:
            if split_same_port:
                self.rtcm_fd = os.open(self.rtcm_serial_port, os.O_WRONLY | os.O_NOCTTY | os.O_NONBLOCK)
                set_serial_raw(self.rtcm_fd, self.rtcm_baud)
                self.get_logger().info(f"Opened RTCM writer on {self.rtcm_serial_port} at {self.rtcm_baud}")
            else:
                self.rtcm_fd = self.nmea_fd
        else:
            self.rtcm_fd = os.open(self.rtcm_serial_port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            set_serial_raw(self.rtcm_fd, self.rtcm_baud)
            self.get_logger().info(f"Opened RTCM serial {self.rtcm_serial_port} at {self.rtcm_baud}")
            self._send_serial_init_commands(self.rtcm_fd, "rtcm_serial_init_commands", self.rtcm_serial_port)

    def _send_serial_init_commands(self, fd: int, param_name: str, port: str):
        commands = str(self.get_parameter(param_name).value or "")
        for command in commands.split(";"):
            command = command.strip()
            if not command:
                continue
            payload = command.encode("ascii") + b"\r\n"
            written = self._write_fd(fd, payload, timeout_s=0.5)
            if written == len(payload):
                self.get_logger().info(f"Sent init command to {port}: {command}")
            else:
                self.get_logger().warning(f"Could not fully send init command to {port}: {command}")

    def _serial_loop(self):
        buf = b""
        max_buf_bytes = 65536
        stale_s = float(self.get_parameter("nmea_stale_restart_s").value)
        # Armed only after the first sentence: a receiver that never talks is a
        # different fault, and arming at startup would loop restarts forever.
        last_rx = None
        while not self.stop_event.is_set():
            # A stalled USB read endpoint (CH340 "urb stopped: -32") is silent:
            # select() neither reports readable nor raises. Exit rather than
            # reopen -- nmea_fd/rtcm_fd share the tty, both must close first.
            if stale_s > 0 and last_rx is not None and time.monotonic() - last_rx > stale_s:
                self.get_logger().fatal(
                    f"No NMEA for {time.monotonic() - last_rx:.0f}s on {self.serial_port} "
                    f"(USB read endpoint stalled?); exiting so the supervisor reopens it"
                )
                os._exit(1)
            try:
                readable, _, _ = select.select([self.nmea_fd], [], [], 0.2)
                if not readable:
                    continue
                data = os.read(self.nmea_fd, 4096)
                if not data:
                    continue
                last_rx = time.monotonic()
            except OSError as exc:
                # Only real I/O errors on the serial fd should back off; a bad
                # NMEA sentence must never stall reading (kernel buffer would
                # overflow and we would lose data).
                self.get_logger().error(f"Serial read error: {exc}")
                time.sleep(1.0)
                continue

            buf += data
            while True:
                cr_pos = buf.find(b"\r")
                lf_pos = buf.find(b"\n")

                if cr_pos == -1 and lf_pos == -1:
                    break

                if cr_pos != -1 and (lf_pos == -1 or cr_pos < lf_pos):
                    split_pos = cr_pos
                else:
                    split_pos = lf_pos

                line = buf[:split_pos]
                buf = buf[split_pos+1:]

                if buf.startswith(b"\r") or buf.startswith(b"\n"):
                    buf = buf[1:]

                line = line.decode("ascii", errors="ignore").strip()
                if line and line[0] in ("$", "#"):
                    try:
                        self._handle_receiver_line(line)
                    except Exception as exc:
                        with self._lock:
                            self.nmea_parse_error_count += 1
                        self.get_logger().warning(f"Failed to parse receiver line {line!r}: {exc}")

            # Drop a runaway buffer that never yields a line terminator so a
            # stream of garbage cannot grow memory without bound.
            if len(buf) > max_buf_bytes:
                self.get_logger().warning(f"Dropping {len(buf)} bytes of unterminated serial data")
                buf = b""

    def _ntrip_loop(self):
        while not self.stop_event.is_set():
            sock = None
            last_sent_gga_source = "none"
            try:
                sock, last_sent_gga_source = self._connect_ntrip()
                initial_data = self._read_http_header(sock)
                sock.settimeout(float(self.get_parameter("ntrip_recv_timeout_s").value))
                with self._lock:
                    self.ntrip_connected = True
                    self.ntrip_connect_count += 1
                last_sent_gga_source = self._send_ntrip_gga(sock)
                last_gga = time.monotonic()
                if initial_data:
                    self._handle_rtcm_data(initial_data)
                while not self.stop_event.is_set():
                    now = time.monotonic()
                    if now - last_gga >= float(self.get_parameter("ntrip_gga_period_s").value):
                        last_sent_gga_source = self._send_ntrip_gga(sock)
                        last_gga = now
                    try:
                        data = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        raise ConnectionError("NTRIP socket closed")
                    self._handle_rtcm_data(data)

            except Exception as exc:
                with self._lock:
                    self.ntrip_connected = False
                    self.ntrip_disconnect_count += 1
                self.get_logger().warning(
                    f"NTRIP disconnected: {exc}; "
                    f"last_gga_source={last_sent_gga_source}, "
                    f"rtcm_bytes={self.rtcm_bytes}, written={self.rtcm_written_bytes}"
                )
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                time.sleep(float(self.get_parameter("ntrip_reconnect_s").value))

    def _handle_rtcm_data(self, data: bytes):
        # The blocking serial write stays outside the lock so it never delays
        # the status timer; only the shared counters are guarded.
        written = self._write_serial(data)
        dropped = len(data) - written
        with self._lock:
            self.rtcm_bytes += len(data)
            self.rtcm_written_bytes += written
            if dropped > 0:
                self.rtcm_dropped_bytes += dropped
                self.rtcm_write_fail_count += 1
            if written > 0:
                self.last_rtcm_time = time.monotonic()
        if dropped > 0:
            self.get_logger().warning(f"Serial write timeout, dropped {dropped}/{len(data)} RTCM bytes")

    def _write_serial(self, data: bytes) -> int:
        if self.rtcm_fd is None:
            return 0
        return self._write_fd(self.rtcm_fd, data, float(self.get_parameter("rtcm_write_timeout_s").value))

    def _write_fd(self, fd: int, data: bytes, timeout_s: float) -> int:
        deadline = time.monotonic() + timeout_s
        written = 0
        while written < len(data) and not self.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            _, writable, _ = select.select([], [fd], [], remaining)
            if not writable:
                break
            try:
                chunk_written = os.write(fd, data[written:])
            except BlockingIOError:
                time.sleep(0.001)
                continue
            except OSError:
                break
            if chunk_written <= 0:
                break
            written += chunk_written
        return written

    def _select_ntrip_gga(self):
        with self._lock:
            live_gga = self.latest_position_gga
        if live_gga:
            return live_gga, "live"
        return self.get_parameter("ntrip_initial_gga").value, "initial"

    def _send_ntrip_gga(self, sock):
        gga, gga_source = self._select_ntrip_gga()
        if gga:
            sock.sendall((gga.strip() + "\r\n").encode("ascii"))
        with self._lock:
            self.latest_ntrip_gga_source = gga_source
        return gga_source

    def _connect_ntrip(self):
        host = self.get_parameter("ntrip_host").value
        port = int(self.get_parameter("ntrip_port").value)
        mount = self.get_parameter("ntrip_mountpoint").value.lstrip("/")
        user = self.get_parameter("ntrip_user").value
        password = self.get_parameter("ntrip_password").value
        if not host or not mount or not user or not password:
            raise ValueError(
                "NTRIP config is incomplete. Set ntrip_host, ntrip_mountpoint, "
                "ntrip_user, and ntrip_password via ROS params or TINYNAV_NTRIP_* env vars."
            )
        auth = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        version = str(self.get_parameter("ntrip_request_version").value)
        if version == "2.0":
            req = (
                f"GET /{mount} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Ntrip-Version: Ntrip/2.0\r\n"
                f"User-Agent: NTRIP TinyNav/1.0\r\n"
                f"Authorization: Basic {auth}\r\n"
                f"Connection: keep-alive\r\n\r\n"
            )
        else:
            req = (
                f"GET /{mount} HTTP/1.0\r\n"
                f"User-Agent: NTRIP TinyNav/1.0\r\n"
                f"Authorization: Basic {auth}\r\n\r\n"
            )
        sock = socket.create_connection((host, port), timeout=10.0)
        sock.settimeout(10.0)
        sock.sendall(req.encode("ascii"))
        gga_source = self._send_ntrip_gga(sock)
        self.get_logger().info(
            f"Opened NTRIP TCP {host}:{port}/{mount} request_version={version}, "
            f"initial_gga_source={gga_source}"
        )
        return sock, gga_source

    def _read_http_header(self, sock):
        data = b""
        max_header_bytes = 8192
        start = time.monotonic()
        while time.monotonic() - start < 10.0:
            try:
                chunk = sock.recv(256)
                if not chunk:
                    header = data.decode("latin1", errors="ignore").strip()
                    raise ConnectionError(f"NTRIP socket closed while reading header: {header!r}")
                data += chunk
                status_line, payload_after_status = self._split_status_line(data)
                if status_line is None:
                    if len(data) > max_header_bytes:
                        raise ConnectionError("NTRIP header too large before status line")
                    continue
                if "200" not in status_line and "ICY" not in status_line:
                    raise ConnectionError(f"Bad NTRIP response: {status_line}")
                if status_line.startswith("ICY"):
                    self.get_logger().info(f"NTRIP header: {status_line!r}")
                    return payload_after_status
                header, payload = self._split_http_header(data)
                if header is not None:
                    self.get_logger().info(f"NTRIP header: {header[:100]!r}")
                    return payload
                if len(data) > max_header_bytes:
                    raise ConnectionError(f"NTRIP HTTP header too large: {status_line}")
            except socket.timeout:
                continue

        header = data.decode("latin1", errors="ignore").strip()
        raise TimeoutError(f"Timed out reading NTRIP header: {header[:100]!r}")

    @staticmethod
    def _split_status_line(data: bytes):
        lf = data.find(b"\n")
        if lf < 0:
            return None, b""
        line = data[: lf + 1].decode("latin1", errors="ignore").strip()
        return line, data[lf + 1 :]

    @staticmethod
    def _split_http_header(data: bytes):
        for sep in (b"\r\n\r\n", b"\n\n"):
            pos = data.find(sep)
            if pos >= 0:
                header = data[:pos].decode("latin1", errors="ignore")
                return header, data[pos + len(sep) :]
        return None, b""

    def _handle_receiver_line(self, line: str):
        # Serialize all receiver-driven state updates so the 1 Hz status timer
        # (a different thread) always reads a coherent snapshot.
        with self._lock:
            if line.startswith("$"):
                self._handle_nmea_line(line)
            elif line.startswith("#"):
                self._handle_unicore_line(line)

    def _handle_nmea_line(self, line: str):
        self.last_nmea_time = time.monotonic()
        self.nmea_sentence_count += 1
        self._mirror_raw_sentence(line)
        has_checksum = "*" in line
        if has_checksum and not nmea_checksum_ok(line):
            self.nmea_checksum_fail_count += 1
            self.get_logger().warning(f"NMEA checksum failed: {line}")
            return
        if not has_checksum and bool(self.get_parameter("strict_nmea_checksum").value):
            self.nmea_checksum_fail_count += 1
            return
        parts = line[1:].split("*")[0].split(",")
        msg_type = parts[0][2:]
        self.latest_sentence = line
        self.latest_sentence_type = msg_type
        if self._should_publish_raw_sentence(msg_type):
            self.raw_pub.publish(String(data=line))
            self.raw_sentence_publish_count += 1
        if msg_type == "GGA":
            self.nmea_gga_count += 1
            self.latest_gga = line
            self.last_gga_time = time.monotonic()
            if len(parts) > 5 and parts[2] and parts[3] and parts[4] and parts[5]:
                self.latest_position_gga = line
            self._parse_gga(parts)
        elif msg_type == "RMC":
            self.nmea_rmc_count += 1
            self._parse_rmc(parts)
        elif msg_type in ("HDT", "THS"):
            self.nmea_heading_count += 1
            self._parse_heading(msg_type, parts)

    def _handle_unicore_line(self, line: str):
        self.last_unicore_log_time = time.monotonic()
        self.unicore_log_count += 1
        self._mirror_raw_sentence(line)
        body = line[1:].split("*", 1)[0]
        if ";" not in body:
            return
        header, payload = body.split(";", 1)
        log_type = header.split(",", 1)[0].upper()
        fields = payload.split(",") if payload else []
        self.latest_unicore_log = line
        self.latest_unicore_log_type = log_type
        if log_type.startswith("BESTNAV"):
            self._parse_bestnav_log(fields)
        elif log_type.startswith("RTKSTATUS"):
            self._parse_rtkstatus_log(fields)
        elif log_type.startswith("RTCMSTATUS"):
            self._parse_rtcmstatus_log(fields)
        elif log_type.startswith("UNILOGLIST"):
            self._parse_uniloglist_log(fields)

    def _parse_bestnav_log(self, fields: list[str]):
        if len(fields) < 10:
            return
        solution_status = fields[0]
        position_type = fields[1]
        lat_std = parse_float_or_none(fields[7])
        lon_std = parse_float_or_none(fields[8])
        hgt_std = parse_float_or_none(fields[9])
        self.latest_bestnav_solution_status = solution_status or None
        self.latest_bestnav_position_type = position_type or None
        self.latest_receiver_position_type = position_type or self.latest_receiver_position_type
        self.latest_receiver_stage = self._select_receiver_stage()
        if lat_std is not None and lon_std is not None and hgt_std is not None:
            self.latest_bestnav_std = {
                "lat_std_m": lat_std,
                "lon_std_m": lon_std,
                "hgt_std_m": hgt_std,
            }
        self.latest_bestnav_station_id = fields[10] if len(fields) > 10 else ""
        self.latest_bestnav_diff_age = parse_float_or_none(fields[11]) if len(fields) > 11 else None
        self.unicore_status_count += 1

    def _parse_rtkstatus_log(self, fields: list[str]):
        if len(fields) < 13:
            return
        position_type = fields[11]
        calculate_status = parse_int_or_none(fields[12])
        self.latest_rtk_position_type = position_type or None
        if position_type:
            self.latest_receiver_position_type = position_type
        self.latest_rtk_calculate_status = calculate_status
        self.latest_rtk_calculate_status_name = (
            None if calculate_status is None else RTK_CALCULATE_STATUS_NAMES.get(calculate_status, "UNKNOWN")
        )
        self.latest_rtk_ion_detected = parse_int_or_none(fields[13]) if len(fields) > 13 else None
        self.latest_rtk_dual_rtk_flag = parse_int_or_none(fields[14]) if len(fields) > 14 else None
        self.latest_receiver_stage = self._select_receiver_stage()
        self.unicore_status_count += 1

    def _parse_rtcmstatus_log(self, fields: list[str]):
        if len(fields) < 4:
            return
        self.latest_rtcm_status = {
            "msg_id": parse_int_or_none(fields[0]),
            "msg_num": parse_int_or_none(fields[1]),
            "base_id": parse_int_or_none(fields[2]),
            "sats_num": parse_int_or_none(fields[3]),
            "l1_num": parse_int_or_none(fields[4]) if len(fields) > 4 else None,
            "l2_num": parse_int_or_none(fields[5]) if len(fields) > 5 else None,
            "l3_num": parse_int_or_none(fields[6]) if len(fields) > 6 else None,
            "l4_num": parse_int_or_none(fields[7]) if len(fields) > 7 else None,
            "l5_num": parse_int_or_none(fields[8]) if len(fields) > 8 else None,
        }
        self.unicore_status_count += 1

    def _parse_uniloglist_log(self, fields: list[str]):
        self.latest_uniloglist = [field for field in fields if field]
        self.unicore_status_count += 1

    def _should_publish_raw_sentence(self, msg_type: str) -> bool:
        return not self.raw_sentence_types or msg_type in self.raw_sentence_types

    def _parse_gga(self, p: list[str]):
        # Fields 0..9 (through altitude) are all we require. Some receivers omit
        # the trailing differential-age / station-id fields; those are already
        # guarded by explicit length checks below, so we must not drop an
        # otherwise valid fix just because the tail is short.
        if len(p) < 10:
            return
        lat = nmea_latlon(p[2], p[3])
        lon = nmea_latlon(p[4], p[5])
        if lat is None or lon is None:
            return
        quality = int(p[6] or "0")
        num_satellites = int(p[7] or "0")
        hdop = float(p[8] or "nan")
        alt = float(p[9] or "0.0")
        undulation = float(p[11] or "0.0") if len(p) > 11 and p[11] else 0.0
        differential_age = float(p[13]) if len(p) > 13 and p[13] else None
        station_id = p[14] if len(p) > 14 else ""
        stamp = ros_time_from_utc(p[1]) or self.get_clock().now().to_msg()
        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = "gps"
        fix.status.status = self._gga_quality_to_status(quality)
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude = lat
        fix.longitude = lon
        fix.altitude = alt + undulation
        sigma_h, sigma_v = self._sigma_from_quality(quality)
        fix.position_covariance = [sigma_h**2, 0.0, 0.0, 0.0, sigma_h**2, 0.0, 0.0, 0.0, sigma_v**2]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        self.fix_pub.publish(fix)
        self.latest_fix = fix
        self.latest_gga_quality = quality
        self.latest_receiver_position_type = (
            self.latest_rtk_position_type
            or self.latest_bestnav_position_type
            or position_type_from_gga_quality(quality)
        )
        self.latest_receiver_stage = self._select_receiver_stage()
        self.last_fix_time = time.monotonic()
        self.latest_num_satellites = num_satellites
        self.latest_hdop = hdop
        self.latest_gga_utc = p[1]
        self.latest_gga_differential_age = differential_age
        self.latest_gga_station_id = station_id
        self._publish_odom_from_fix(fix)

        time_ref = TimeReference()
        time_ref.header = fix.header
        time_ref.time_ref = stamp
        time_ref.source = "nmea_gga"
        self.latest_time_reference = stamp
        self.time_ref_pub.publish(time_ref)

    def _parse_rmc(self, p: list[str]):
        if len(p) < 10 or p[2] != "A":
            return
        stamp = ros_time_from_utc(p[1], p[9]) or self.get_clock().now().to_msg()
        speed = float(p[7] or "0.0") * KNOT_TO_MPS
        course = math.radians(float(p[8] or "0.0"))
        yaw = self._wrap_angle(math.pi / 2.0 - course)
        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.header.frame_id = self.frame_id
        twist.twist.linear.x = speed * math.cos(yaw)
        twist.twist.linear.y = speed * math.sin(yaw)
        twist.twist.linear.z = 0.0
        self.latest_velocity = np.array([twist.twist.linear.x, twist.twist.linear.y, 0.0], dtype=np.float64)
        self.latest_velocity_stamp = stamp
        self.vel_pub.publish(twist)

    def _parse_heading(self, msg_type: str, p: list[str]):
        if not self.use_heading:
            # Heading disabled: do not publish /heading and do not update the
            # yaw used by /rtk/odom. THS/HDT are still counted/mirrored upstream.
            return
        if len(p) < 2 or not p[1]:
            return
        # THS carries a status field: 'A'=valid, anything else (e.g. 'V') means
        # the heading solution is unavailable and must not be published. HDT has
        # no such flag, so it is only length/value checked above.
        if msg_type == "THS" and (len(p) < 3 or p[2].strip().upper() != "A"):
            return
        heading = math.radians(float(p[1]))
        yaw = self._wrap_angle(math.pi / 2.0 - heading + self.heading_offset)
        self.latest_heading_yaw = yaw
        self.latest_heading_stamp = self.get_clock().now().to_msg()
        quat = yaw_to_quat(yaw)
        msg = QuaternionStamped()
        msg.header.stamp = self.latest_heading_stamp
        msg.header.frame_id = self.frame_id
        msg.quaternion.x = float(quat[0])
        msg.quaternion.y = float(quat[1])
        msg.quaternion.z = float(quat[2])
        msg.quaternion.w = float(quat[3])
        self.heading_pub.publish(msg)

    def _publish_odom_from_fix(self, msg: NavSatFix):
        if msg.status.status < self.min_navsat_status:
            self.latest_enu = None
            return
        if self.origin is None:
            if msg.status.status < self.origin_min_navsat_status:
                # Hold off publishing odom until we can lock a clean origin.
                self.latest_enu = None
                return
            self.origin = make_origin(msg.latitude, msg.longitude, msg.altitude)
            self.get_logger().info(
                f"Initialized RTK ENU origin lat={msg.latitude:.9f}, "
                f"lon={msg.longitude:.9f}, alt={msg.altitude:.3f}"
            )
        position = lla_to_enu(msg.latitude, msg.longitude, msg.altitude, self.origin)
        self.latest_enu = [float(v) for v in position]
        yaw = self.latest_heading_yaw if self.use_heading else 0.0
        quat = yaw_to_quat(yaw)
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = float(position[0])
        odom.pose.pose.position.y = float(position[1])
        odom.pose.pose.position.z = float(position[2])
        odom.pose.pose.orientation.x = float(quat[0])
        odom.pose.pose.orientation.y = float(quat[1])
        odom.pose.pose.orientation.z = float(quat[2])
        odom.pose.pose.orientation.w = float(quat[3])
        odom.twist.twist.linear.x = float(self.latest_velocity[0])
        odom.twist.twist.linear.y = float(self.latest_velocity[1])
        odom.twist.twist.linear.z = float(self.latest_velocity[2])
        self._copy_position_covariance(msg, odom)
        self.odom_pub.publish(odom)
        self._publish_path(odom)
        if self.publish_tf:
            self.tf_broadcaster.sendTransform(self._odom_to_tf(odom))

    def _load_origin_from_params(self):
        lat = float(self.get_parameter("origin_lat").value)
        lon = float(self.get_parameter("origin_lon").value)
        alt = float(self.get_parameter("origin_alt").value)
        if math.isfinite(lat) and math.isfinite(lon) and math.isfinite(alt):
            origin = make_origin(lat, lon, alt)
            self.get_logger().info(f"Using configured RTK origin lat={lat:.9f}, lon={lon:.9f}, alt={alt:.3f}")
            return origin
        return None

    def _copy_position_covariance(self, fix_msg: NavSatFix, odom_msg: Odometry):
        cov = list(odom_msg.pose.covariance)
        fix_cov = list(fix_msg.position_covariance)
        cov[0] = fix_cov[0]
        cov[1] = fix_cov[1]
        cov[2] = fix_cov[2]
        cov[6] = fix_cov[3]
        cov[7] = fix_cov[4]
        cov[8] = fix_cov[5]
        cov[12] = fix_cov[6]
        cov[13] = fix_cov[7]
        cov[14] = fix_cov[8]
        cov[35] = 0.1 if self.use_heading else 999.0
        odom_msg.pose.covariance = cov

    def _publish_path(self, odom: Odometry):
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.path.header.stamp = odom.header.stamp
        self.path.poses.append(pose)
        if len(self.path.poses) > self.path_max_size:
            self.path.poses = self.path.poses[-self.path_max_size:]
        self.path_pub.publish(self.path)

    def _publish_status_timer(self):
        # Hold the lock across the whole snapshot so every field in the three
        # status messages comes from the same instant, never a torn mix of
        # values updated by the serial/NTRIP threads mid-build.
        with self._lock:
            if self.latest_fix is None:
                self._publish_status(None, accepted=False, position=None)
            else:
                accepted = self.latest_fix.status.status >= self.min_navsat_status and not self._fix_is_stale()
                self._publish_status(self.latest_fix, accepted=accepted, position=self.latest_enu)

    def _fix_is_stale(self) -> bool:
        # Age of the last GGA that carried a POSITION, not of the last sentence:
        # a receiver losing lock keeps emitting empty GGA, which kept this False
        # while the reported fix was already minutes old.
        if self.last_fix_time is None:
            return True
        return time.monotonic() - self.last_fix_time > float(self.get_parameter("fix_stale_after_s").value)

    def _reported_fix_state(self):
        # quality/stage/position_type describe one fix, so they must not outlive
        # it -- a dead link otherwise reads as a healthy DGNSS fix forever.
        if self._fix_is_stale():
            return 0, stage_from_gga_quality(0), None
        return self.latest_gga_quality, self.latest_receiver_stage, self.latest_receiver_position_type

    def _select_receiver_stage(self) -> str:
        for position_type in (
            self.latest_rtk_position_type,
            self.latest_bestnav_position_type,
            self.latest_receiver_position_type,
        ):
            stage = stage_from_position_type(position_type)
            if stage is not None:
                return stage
        return stage_from_gga_quality(self.latest_gga_quality)

    def _receiver_status_payload(self, msg: NavSatFix | None, accepted: bool, position):
        navsat_status = None if msg is None else int(msg.status.status)
        quality, stage, position_type = self._reported_fix_state()
        return {
            "seq": self.status_seq,
            "accepted": accepted,
            "receiver_stage": stage,
            "gga_quality": quality,
            "gga_quality_name": gga_quality_name(quality),
            "ros_navsat_status": navsat_status,
            "ros_navsat_status_name": navsat_status_name(navsat_status),
            "receiver_position_type": position_type,
            "bestnav_solution_status": self.latest_bestnav_solution_status,
            "bestnav_position_type": self.latest_bestnav_position_type,
            "bestnav_std": self.latest_bestnav_std,
            "bestnav_diff_age_s": self.latest_bestnav_diff_age,
            "bestnav_station_id": self.latest_bestnav_station_id or None,
            "rtk_position_type": self.latest_rtk_position_type,
            "rtk_calculate_status": self.latest_rtk_calculate_status,
            "rtk_calculate_status_name": self.latest_rtk_calculate_status_name,
            "rtk_ion_detected": self.latest_rtk_ion_detected,
            "rtk_dual_rtk_flag": self.latest_rtk_dual_rtk_flag,
            "rtcm_status": self.latest_rtcm_status,
            "latest_unicore_log_type": self.latest_unicore_log_type or None,
            "unicore_log_count": self.unicore_log_count,
            "unicore_status_count": self.unicore_status_count,
            "position": position,
        }

    def _publish_status(self, msg: NavSatFix | None, accepted: bool, position):
        now = time.monotonic()
        self.status_seq += 1
        nmea_age = None if self.last_nmea_time is None else now - self.last_nmea_time
        gga_age = None if self.last_gga_time is None else now - self.last_gga_time
        fix_age = None if self.last_fix_time is None else now - self.last_fix_time
        rtcm_age = None if self.last_rtcm_time is None else now - self.last_rtcm_time
        quality, stage, position_type = self._reported_fix_state()
        io_status = {
            "seq": self.status_seq,
            "ntrip_connected": self.ntrip_connected,
            "last_nmea_age_s": nmea_age,
            "last_gga_age_s": gga_age,
            "last_rtcm_age_s": rtcm_age,
            "latest_sentence_type": self.latest_sentence_type or None,
            "nmea_sentence_count": self.nmea_sentence_count,
            "nmea_gga_count": self.nmea_gga_count,
            "nmea_rmc_count": self.nmea_rmc_count,
            "nmea_heading_count": self.nmea_heading_count,
            "raw_sentence_publish_count": self.raw_sentence_publish_count,
            "nmea_checksum_fail_count": self.nmea_checksum_fail_count,
            "nmea_parse_error_count": self.nmea_parse_error_count,
            "rtcm_bytes": self.rtcm_bytes,
            "rtcm_written_bytes": self.rtcm_written_bytes,
            "rtcm_dropped_bytes": self.rtcm_dropped_bytes,
            "rtcm_write_fail_count": self.rtcm_write_fail_count,
            "raw_pty_dropped_lines": self.raw_pty_dropped_lines,
            "ntrip_gga_source": self.latest_ntrip_gga_source,
            "gga_quality": quality,
            "gga_quality_name": gga_quality_name(quality),
            "receiver_stage": stage,
            "receiver_position_type": position_type,
            "unicore_log_count": self.unicore_log_count,
            "fix_stale": self._fix_is_stale(),
        }
        navsat_status = None if msg is None else int(msg.status.status)
        status = {
            "seq": self.status_seq,
            "accepted": accepted,
            "navsat_status": navsat_status,
            "navsat_status_name": navsat_status_name(navsat_status),
            "service": None if msg is None else int(msg.status.service),
            "fix_stale": self._fix_is_stale(),
            "last_gga_age_s": gga_age,
            # Age of the last POSITIONED GGA; last_gga_age_s counts empty ones too.
            "fix_age_s": fix_age,
            "last_rtcm_age_s": rtcm_age,
            "gga_quality": quality,
            "gga_quality_name": gga_quality_name(quality),
            "receiver_stage": stage,
            "receiver_position_type": position_type,
            "bestnav_solution_status": self.latest_bestnav_solution_status,
            "bestnav_position_type": self.latest_bestnav_position_type,
            "bestnav_std": self.latest_bestnav_std,
            "rtk_calculate_status": self.latest_rtk_calculate_status,
            "rtk_calculate_status_name": self.latest_rtk_calculate_status_name,
            "rtcm_status": self.latest_rtcm_status,
            "gga_utc": self.latest_gga_utc or None,
            "gga_differential_age_s": self.latest_gga_differential_age,
            "gga_station_id": self.latest_gga_station_id or None,
            "num_satellites": self.latest_num_satellites,
            "hdop": None if not math.isfinite(self.latest_hdop) else self.latest_hdop,
            "latitude": None if msg is None else msg.latitude,
            "longitude": None if msg is None else msg.longitude,
            "altitude": None if msg is None else msg.altitude,
            "enu": position,
            "latest_gga": self.latest_gga or None,
            "origin_ready": self.origin is not None,
            "heading_ready": self.latest_heading_stamp is not None,
            "velocity_ready": self.latest_velocity_stamp is not None,
        }
        receiver_status = self._receiver_status_payload(msg, accepted, position)
        self.status_pub.publish(String(data=json.dumps(status, separators=(",", ":"))))
        self.io_status_pub.publish(String(data=json.dumps(io_status, separators=(",", ":"))))
        self.receiver_status_pub.publish(String(data=json.dumps(receiver_status, separators=(",", ":"))))

    def _odom_to_tf(self, odom: Odometry):
        from geometry_msgs.msg import TransformStamped

        tf_msg = TransformStamped()
        tf_msg.header = odom.header
        tf_msg.child_frame_id = odom.child_frame_id
        tf_msg.transform.translation.x = odom.pose.pose.position.x
        tf_msg.transform.translation.y = odom.pose.pose.position.y
        tf_msg.transform.translation.z = odom.pose.pose.position.z
        tf_msg.transform.rotation = odom.pose.pose.orientation
        return tf_msg

    @staticmethod
    def _sigma_from_quality(quality: int) -> tuple[float, float]:
        # Conservative per-solution-type placeholders (horizontal, vertical) in
        # metres until real per-epoch stddev is wired in from GPGST/BESTNAVA.
        # RTK float (5) must NOT share RTK fixed (4) optimism.
        return {
            4: (0.02, 0.08),   # RTK fixed
            5: (0.15, 0.30),   # RTK float
            2: (0.50, 1.00),   # DGNSS
            1: (2.50, 5.00),   # single point
        }.get(int(quality), (2.50, 5.00))

    @staticmethod
    def _gga_quality_to_status(quality: int) -> int:
        if quality in (4, 5):
            return NavSatStatus.STATUS_GBAS_FIX
        if quality > 0:
            return NavSatStatus.STATUS_FIX
        return NavSatStatus.STATUS_NO_FIX

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _parse_sentence_type_filter(value):
        text = str(value or "").strip().upper()
        if not text or text in ("*", "ALL"):
            return set()
        return {part.strip() for part in text.split(",") if part.strip()}

    def destroy_node(self):
        self.stop_event.set()
        for fd in {self.nmea_fd, self.rtcm_fd}:
            if fd is None:
                continue
            try:
                os.close(fd)
            except OSError:
                pass
        if self.raw_pty_master is not None:
            try:
                os.close(self.raw_pty_master)
            except OSError:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RtkBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
