"""One logging setup for every node in the stack -- pilot included.

Each node writes its own file under `<db>/logs/<YYYY-MM-DD>/<tag>.log` through
stdlib logging; this module is the single place that knows the layout, the
format, the rotation rule and the retention rule, so the nodes only decide
*their tag*:

    from tinynav.core.logsetup import setup_logging
    log = setup_logging('planning')

Stdlib only -- no rclpy, no numpy -- so any script on any side (pilot's, the
fork's, an editor) can import it before ROS exists.

Rules, all here and nowhere else:
  one file per tag per day, appended by at most one process at a time
  (a tag maps to one running node; two same-tag processes share a file and
  deserve the tangle they get);
  the file rolls at midnight into the new day's directory;
  day directories older than RETENTION_DAYS are swept -- at process start and
  then by a daemon thread every SWEEP_PERIOD_SEC, because a rig runs for
  months without a restart;
  timestamps are RFC3339 local with milliseconds; the console copy (stdout,
  i.e. docker logs where a container owns the pid) carries the same format at
  console_level.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from datetime import datetime, timedelta

#: RFC3339-ish local time. The rigs carry /etc/localtime; the sim box does too.
_FORMAT = '%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s'
_DATEFMT = '%Y-%m-%dT%H:%M:%S'

#: The one retention number. Day directories older than this are removed.
RETENTION_DAYS = 14

SWEEP_PERIOD_SEC = 24 * 3600

_DAY_RE = re.compile(r'\d{4}-\d{2}-\d{2}')

#: Guards against two tee installs in one process.
_console_captured = False

_sweeper_lock = threading.Lock()
_sweeper_started = False


def db_root() -> str:
    return os.environ.get('TINYNAV_DB_PATH', '/tinynav/tinynav_db')


def logs_root() -> str:
    return os.path.join(db_root(), 'logs')


def _today() -> str:
    return time.strftime('%Y-%m-%d')


def every(period_s: float):
    """A pass-once-per-period gate for N Hz status lines: `if due(): log.info(...)`.

    A call-site gate rather than a logging.Filter, which still builds the record
    it drops. Fires immediately on first call.
    """
    next_due = 0.0

    def due() -> bool:
        nonlocal next_due
        now = time.monotonic()
        if now < next_due:
            return False
        next_due = now + period_s
        return True

    return due


class DayDirFileHandler(logging.Handler):
    """`<root>/<YYYY-MM-DD>/<tag>.log`, reopened when the date rolls over.

    Not logging.handlers.TimedRotatingFileHandler: that one rotates the file
    but keeps writing into the old directory's name, which is exactly the
    per-day layout this exists to provide. emit() runs under the handler lock,
    and one tag has one writer, so plain append is safe.
    """

    def __init__(self, tag: str, root: str | None = None,
                 retention_days: int = RETENTION_DAYS):
        super().__init__()
        self._tag = tag
        self._root = root or logs_root()
        self._retention_days = retention_days
        self._day = ''
        self._fh = None

    def emit(self, record):
        # The stdlib handler contract: a malformed record (a call site whose
        # args do not match its format) must degrade to handleError -- printed
        # to stderr, line dropped -- and never propagate into the caller. The
        # first unguarded version here let one bad debug() kill the VIO loop
        # on every frame (perception_node.py:407, 1847 consecutive failures).
        try:
            day = _today()
            if self._fh is None or day != self._day:
                if self._fh is not None:
                    self._fh.close()
                directory = os.path.join(self._root, day)
                os.makedirs(directory, exist_ok=True)
                self._fh = open(os.path.join(directory, f'{self._tag}.log'),
                                'a', encoding='utf-8')
                self._day = day
            self._fh.write(self.format(record) + '\n')
            self._fh.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        super().close()


def sweep(root: str | None = None, retention_days: int = RETENTION_DAYS) -> list:
    """Delete day directories older than the retention. Returns what it removed."""
    root = root or logs_root()
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime('%Y-%m-%d')
    removed = []
    try:
        names = os.listdir(root)
    except OSError:
        return removed
    for name in names:
        if not _DAY_RE.fullmatch(name) or name >= cutoff:
            continue
        shutil.rmtree(os.path.join(root, name), ignore_errors=True)
        removed.append(name)
    return removed


def _ensure_sweeper(root: str, retention_days: int) -> None:
    """One sweep thread per process, however many loggers call setup."""
    global _sweeper_started
    with _sweeper_lock:
        if _sweeper_started:
            return
        _sweeper_started = True

    def _loop():
        while True:
            sweep(root, retention_days)
            time.sleep(SWEEP_PERIOD_SEC)

    threading.Thread(target=_loop, daemon=True, name='log-retention').start()


def setup_logging(tag: str, *, console_level: int = logging.INFO,
                  retention_days: int = RETENTION_DAYS,
                  root: str | None = None) -> logging.Logger:
    """The one entry point. Reconfigures rather than duplicates on a second call."""
    logger = logging.getLogger(tag)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    file_handler = DayDirFileHandler(tag, root, retention_days)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    _ensure_sweeper(root or logs_root(), retention_days)
    return logger


def capture_console(root: str | None = None,
                    retention_days: int = RETENTION_DAYS) -> bool:
    """Tee the process's raw stdout+stderr into `<root>/<day>/console.log`.

    The copy of what docker logs would show, on disk and rotating: child
    processes that inherit these fds are captured too, including anything they
    print before (or instead of) their own setup_logging -- a pre-bootstrap
    traceback, a bare print, a library that insists on stdout. Must be called
    before any children are spawned. Once per process.
    """
    global _console_captured
    if _console_captured:
        return False
    _console_captured = True
    root = root or logs_root()
    try:
        out = os.dup(1)
        read_fd, write_fd = os.pipe()
        # One merged stream: fd1 and fd2 both point at the pipe, so the pump
        # forwards through `out` alone -- writing both originals would print
        # every line twice wherever they land on the same pane.
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
    except OSError:
        return False

    def _write_all(fd, data):
        while data:
            try:
                data = data[os.write(fd, data):]
            except OSError:
                return

    state = {'day': '', 'fh': None}

    def _file_for(day):
        if state['fh'] is not None and day == state['day']:
            return state['fh']
        if state['fh'] is not None:
            try:
                state['fh'].close()
            except OSError:
                pass
        directory = os.path.join(root, day)
        os.makedirs(directory, exist_ok=True)
        state['fh'] = open(os.path.join(directory, 'console.log'), 'ab')
        state['day'] = day
        return state['fh']

    def _pump():
        while True:
            try:
                data = os.read(read_fd, 65536)
            except OSError:
                return
            if not data:
                return
            _write_all(out, data)
            try:
                # `data` is raw bytes and the file is binary -- a decode here
                # would raise TypeError inside a non-OSError guard and kill
                # this thread, which is what stops the whole process the next
                # time the pipe fills.
                _file_for(_today()).write(data)
                state['fh'].flush()
            except Exception:
                pass

    threading.Thread(target=_pump, daemon=True, name='console-tee').start()
    return True
