"""Checks for docker/jetson-lean: the drift check catches what it should, and the host
check accepts a supported Jetson and names what is wrong with any other.

    python3 -m unittest discover -s docker/jetson-lean -p 'test_*.py'
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import check_sync

HERE = Path(__file__).resolve().parent
ROOT_DOCKERFILE = (HERE.parents[1] / 'Dockerfile').read_text()
LEAN = (HERE / 'Dockerfile').read_text()
ENTRYPOINT = (HERE / 'entrypoint.sh').read_text()


class Sync(unittest.TestCase):
    def problems(self, root=ROOT_DOCKERFILE, lean=LEAN, entrypoint=ENTRYPOINT):
        return check_sync.check(root, lean, entrypoint)

    def test_the_lean_dockerfile_follows_the_root_one(self):
        self.assertEqual(self.problems(), [])

    def test_a_changed_or_new_root_step_is_drift(self):
        changed = ROOT_DOCKERFILE.replace('ros-humble-foxglove-bridge', 'ros-humble-foxglove-bridge ros-humble-rviz2')
        self.assertTrue(any('not in the lean Dockerfile' in p for p in self.problems(root=changed)))
        added = ROOT_DOCKERFILE.replace('# clean\n', 'RUN pip3 install something-new\n# clean\n')
        self.assertTrue(any('something-new' in p for p in self.problems(root=added)))

    def test_changes_inside_a_listed_difference_are_caught(self):
        trt = ROOT_DOCKERFILE.replace('tensorrt=10.3.0.30-1+cuda12.5', 'tensorrt=10.3.0.31-1+cuda12.5')
        self.assertTrue(any('TensorRT' in p for p in self.problems(root=trt)))
        gtsam = ROOT_DOCKERFILE.replace('-b yzf/add_smart_factor_python_export', '-b other-branch')
        self.assertTrue(any('gtsam' in p for p in self.problems(root=gtsam)))
        rosdep = ROOT_DOCKERFILE.replace('rosdep update', 'rosdep update --include-eol-distros')
        self.assertTrue(self.problems(root=rosdep))
        flutter = ROOT_DOCKERFILE.replace('flutter build web --release', 'flutter build web --release --wasm')
        self.assertTrue(any('web app' in p for p in self.problems(root=flutter)))

    def test_the_entrypoint_body_must_match(self):
        changed = ROOT_DOCKERFILE.replace('It may take 5–10 minutes.', 'It may take 10 minutes.')
        self.assertTrue(any('entrypoint.sh' in p for p in self.problems(root=changed)))
        no_check = check_sync.strip_gpu_check(ENTRYPOINT)
        self.assertNotIn('TINYNAV_SKIP_GPU_CHECK', no_check)

    def test_order_matters(self):
        a = 'RUN echo "source /3rdparty/ros2_ws/install/local_setup.bash" >> ~/.bashrc\n'
        b = "RUN echo 'setw -g mouse on' > /root/.tmux.conf\n"
        self.assertIn(a + b, LEAN)
        self.assertTrue(any('order' in p for p in self.problems(lean=LEAN.replace(a + b, b + a))))


class HostCheck(unittest.TestCase):
    """host_check.sh against a fake host tree (TINYNAV_HOST_ROOT) and a fake dpkg-query."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'host'
        lib = self.root / 'usr/lib/aarch64-linux-gnu'
        lib.mkdir(parents=True)
        for name in ('libnvinfer.so', 'libnvinfer.so.10', 'libnvinfer.so.10.3.0', 'libnvinfer_plugin.so.10',
                     'libnvinfer_builder_resource.so.10.3.0', 'libnvonnxparser.so.10', 'libcudnn.so.9',
                     'libcudnn_ops.so.9', 'libcudnn_engines_precompiled.so.9.3.0', 'libcudart.so.12',
                     'libnvinfer.so.10.bak-old'):
            (lib / name).touch()
        (self.root / 'usr/local/cuda-12.6').mkdir(parents=True)
        (self.root / 'etc').mkdir()
        self.release('# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: x')
        self.dpkg = Path(self.tmp.name) / 'dpkg-query'
        self.trt('10.3.0.30-1+cuda12.5')

    def tearDown(self):
        self.tmp.cleanup()

    def release(self, line):
        (self.root / 'etc/nv_tegra_release').write_text(line + '\n')

    def trt(self, version):
        self.dpkg.write_text('#!/bin/sh\nprintf "%s" "{}"\n'.format(version) if version else '#!/bin/sh\nexit 1\n')
        self.dpkg.chmod(0o755)

    def run_check(self, *args, arch='aarch64'):
        env = dict(os.environ, TINYNAV_HOST_ROOT=str(self.root), DPKG_QUERY=str(self.dpkg), TINYNAV_ARCH=arch)
        return subprocess.run(['bash', str(HERE / 'host_check.sh'), *args], capture_output=True, text=True, env=env)

    def test_a_supported_jetson(self):
        result = self.run_check('--no-docker')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('✅ L4T R36.4.3', result.stdout)
        self.assertEqual(self.run_check('--build-args').stdout.split(),
                         ['--build-arg', 'TRT_VERSION=10.3.0.30-1+cuda12.5', '--build-arg', 'CUDA_VERSION=12.6'])

    def test_mounts_are_every_tensorrt_and_cudnn_library_read_only(self):
        mounts = self.run_check('--mounts').stdout.split()
        lib = '/usr/lib/aarch64-linux-gnu/'
        self.assertEqual(mounts[0], '/usr/local/cuda-12.6:/usr/local/cuda-12.6:ro')
        names = sorted(m.split(':')[0].removeprefix(lib) for m in mounts[1:])
        self.assertEqual(names, ['libcudnn.so.9', 'libcudnn_engines_precompiled.so.9.3.0', 'libcudnn_ops.so.9',
                                 'libnvinfer.so', 'libnvinfer.so.10', 'libnvinfer.so.10.3.0',
                                 'libnvinfer_builder_resource.so.10.3.0', 'libnvinfer_plugin.so.10',
                                 'libnvonnxparser.so.10'])
        self.assertTrue(all(m.endswith(':ro') and str(self.root) not in m for m in mounts))

    def test_unsupported_hosts_say_why(self):
        cases = [
            (lambda: self.release('# R35 (release), REVISION: 6.0, GCID: 1'), 'needs R36.4'),
            (lambda: self.release('# R36 (release), REVISION: 3.0, GCID: 1'), 'needs R36.4'),
            (lambda: self.trt('8.6.2.3-1+cuda12.2'), 'needs 10.3.x'),
            (lambda: self.trt(''), 'libnvinfer10) is not installed'),
            (lambda: (self.root / 'usr/lib/aarch64-linux-gnu/libcudnn.so.9').unlink(), 'no /usr/lib/aarch64-linux-gnu/libcudnn.so.9'),
            (lambda: (self.root / 'usr/local/cuda-12.6').rmdir(), 'CUDA 12.6'),
            (lambda: (self.root / 'etc/nv_tegra_release').unlink(), 'not a Jetson'),
        ]
        for spoil, said in cases:
            self.tearDown()
            self.setUp()
            spoil()
            result = self.run_check('--no-docker')
            self.assertEqual(result.returncode, 1, said)
            self.assertIn(said, result.stdout)
            self.assertEqual(self.run_check('--mounts').returncode, 1)
            self.assertEqual(self.run_check('--mounts').stdout, '')  # nothing to mount on a bad host
        result = self.run_check('--no-docker', arch='x86_64')
        self.assertIn('not a Jetson (aarch64)', result.stdout)


if __name__ == '__main__':
    unittest.main()
