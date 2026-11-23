#!/usr/bin/env python3
"""Does docker/jetson-lean/Dockerfile still follow the root Dockerfile?

The lean image is the root Dockerfile's recipe on ros:humble with the host's JetPack
libraries. Every instruction of the root Dockerfile must appear in the lean one, in the
same order, unchanged, except the few differences listed in DIFFERENCES, each of which
is checked exactly (a change upstream inside one of them is caught too). The
entrypoint the root Dockerfile writes must equal docker/jetson-lean/entrypoint.sh with
its GPU check block removed.

    python3 docker/jetson-lean/check_sync.py        # from the repository root; exit 1 on drift
"""
import argparse
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def instructions(text):
    """The Dockerfile's instructions, one string each: continuations joined, comment
    lines dropped, whitespace collapsed; a heredoc is kept with its body."""
    out, lines, i = [], text.split('\n'), 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith('#'):
            continue
        parts = [line]
        while parts[-1].endswith('\\') and i < len(lines):
            parts[-1] = parts[-1][:-1]
            nxt = lines[i].strip()
            i += 1
            if not nxt.startswith('#'):
                parts.append(nxt)
        joined = ' '.join(p.strip() for p in parts)
        heredoc = re.search(r"<<-?'?(\w+)'?\s*$", joined)
        if heredoc:
            body = []
            while i < len(lines) and lines[i] != heredoc.group(1):
                body.append(lines[i])
                i += 1
            i += 1
            out.append(' '.join(joined.split()) + '\n' + '\n'.join(body))
        else:
            out.append(' '.join(joined.split()))
    return out


def heredoc_body(instruction):
    return instruction.split('\n', 1)[1] + '\n'


# Each difference: what it is, the root instruction it applies to (regex), and the lean
# instruction it must become (a function of the root one), or None when the lean file
# does without it; `check(root_instruction, lean_text)` adds a condition. Keep the
# reasons here and in the Dockerfile's header in step.
DIFFERENCES = [
    ('base image: ros:humble instead of the JetPack base',
     r'^FROM ', None, lambda u, lean: 'FROM ${BASE}' in lean and 'ARG BASE=ros:humble' in lean),
    ('this image is aarch64 only',
     r'^ENV ARCH=\$ARCH$', lambda u: 'ENV ARCH=aarch64', None),
    ('this image is aarch64 only',
     r'^RUN echo "ARCH is \$ARCH"$', None, None),
    ('TensorRT: binding and trtexec only, at the version the root Dockerfile installs on aarch64',
     r'^RUN if \[ "\$ARCH" = "x86_64" \]; then .* tensorrt=', None,
     lambda u, lean: re.search(r'aarch64.*?tensorrt=(\S+)', u).group(1)
     == re.search(r'ARG TRT_VERSION=(\S+)', lean).group(1)),
    ("ROS's apt key: already in ros:humble",
     r'^RUN curl -sSL https://raw\.githubusercontent\.com/ros/rosdistro/master/ros\.key ', None, None),
    ("ROS's apt source: already in ros:humble (a second entry for it conflicts)",
     r'^RUN echo "deb \[arch=\$\(dpkg --print-architecture\) signed-by=/usr/share/keyrings/ros-archive-keyring\.gpg\] '
     r'http://packages\.ros\.org/ros2/ubuntu ', None, None),
    ('the build runs as root: no sudo',
     r'^RUN echo "deb .*gazebo/ubuntu-stable .*\| sudo tee ', lambda u: u.replace('| sudo tee ', '| tee '), None),
    ('ros:humble has no /etc/udev/rules.d yet',
     r'^RUN git clone https://github\.com/IntelRealSense/librealsense\.git ',
     lambda u: u.replace('&& cp config/99-realsense-libusb.rules',
                         '&& mkdir -p /etc/udev/rules.d && cp config/99-realsense-libusb.rules'), None),
    ('rosdep is already initialised in ros:humble',
     r'^RUN mkdir -p ros2_ws/src .* rosdep init ',
     lambda u: u.replace('&& rosdep init &&', '&& (rosdep init || true) &&'), None),
    ('the gtsam branch is a build argument with the same default',
     r'^RUN git clone https://github\.com/dvorak0/gtsam\.git -b ',
     lambda u: re.sub(r'-b \S+', '-b ${GTSAM_REF}', u, count=1),
     lambda u, lean: re.search(r'gtsam\.git -b (\S+)', u).group(1)
     == re.search(r'ARG GTSAM_REF=(\S+)', lean).group(1)),
    ('the entrypoint is entrypoint.sh (same body, after a GPU check)',
     r"^RUN cat > /usr/local/bin/entrypoint\.sh <<'EOF'", None, None),
    ('the web app is optional (WITH_APP), with the same commands',
     r'^RUN git clone https://github\.com/flutter/flutter\.git ', None,
     lambda u, lean: u[len('RUN '):] in lean),
    ('the web app is optional (WITH_APP), with the same commands',
     r'^RUN flutter config ', None,
     lambda u, lean: u[len('RUN '):] in lean),
]


def strip_gpu_check(entrypoint):
    """entrypoint.sh without its `# >>> jetson-lean` ... `# <<< jetson-lean` block and
    the blank line before it."""
    return re.sub(r'\n\n# >>> jetson-lean.*?# <<< jetson-lean\n', '\n', entrypoint, flags=re.S)


def check(root_text, lean_text, entrypoint_text):
    problems = []
    root, lean = instructions(root_text), instructions(lean_text)
    lean_joined = '\n'.join(lean)  # what the conditions read: continuations joined
    expected, used = [], set()
    for u in root:
        head = u.split('\n', 1)[0]
        if u in lean:
            expected.append(u)
            continue
        matches = [d for d in DIFFERENCES if re.search(d[1], head)]
        if len(matches) != 1:
            problems.append('root instruction not in the lean Dockerfile (update docker/jetson-lean/Dockerfile, '
                            'or list the difference in check_sync.py):\n    ' + head[:300])
            continue
        reason, _, become, condition = matches[0]
        used.add(matches[0][1])
        if condition and not condition(u, lean_joined):
            problems.append('%s: the root Dockerfile changed here; follow it:\n    %s' % (reason, head[:300]))
        if become is not None:
            want = become(u)
            if want not in lean:
                problems.append('%s: expected in the lean Dockerfile:\n    %s' % (reason, want[:300]))
            else:
                expected.append(want)
        if head.startswith("RUN cat > /usr/local/bin/entrypoint.sh"):
            if strip_gpu_check(entrypoint_text) != heredoc_body(u):
                problems.append('entrypoint.sh (without its GPU check) differs from the entrypoint the root '
                                'Dockerfile writes; copy the new body into docker/jetson-lean/entrypoint.sh')
    for d in DIFFERENCES:
        if d[1] not in used:
            problems.append('listed difference no longer applies (remove it from check_sync.py): ' + d[0])
    # Same order: each one after the one before (an instruction may repeat).
    at = -1
    for e in expected:
        try:
            at = lean.index(e, at + 1)
        except ValueError:
            problems.append('order differs from the root Dockerfile at:\n    ' + e.split('\n', 1)[0][:200])
            break
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=str(ROOT / 'Dockerfile'))
    parser.add_argument('--lean', default=str(HERE / 'Dockerfile'))
    parser.add_argument('--entrypoint', default=str(HERE / 'entrypoint.sh'))
    args = parser.parse_args()
    problems = check(Path(args.root).read_text(), Path(args.lean).read_text(), Path(args.entrypoint).read_text())
    for p in problems:
        print('❌ ' + p)
    if problems:
        sys.exit(1)
    print('✅ docker/jetson-lean follows the root Dockerfile')


if __name__ == '__main__':
    main()
