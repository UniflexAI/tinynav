# TinyNav lean image for Jetson (reuses the host's JetPack)

The root `Dockerfile` builds TinyNav on a JetPack base image, which carries CUDA, cuDNN
and the TensorRT libraries inside the image. A Jetson running JetPack already has the
very same libraries on the host. This image leaves them out and mounts the host's
copies read-only when the container starts. Everything else is the root Dockerfile's
recipe, step by step.

| | Root image | Lean image |
|---|---|---|
| CUDA, cuDNN, TensorRT libraries | inside the image | from the host, mounted read-only |
| TensorRT Python binding, `trtexec` | inside | inside, pinned to the host's exact version |
| Base | JetPack base image | `ros:humble` |
| Web app (Flutter) | inside | inside by default; `--no-app` leaves it out |
| Platforms | x86_64 and Jetson | Jetson only |

Size, measured without the web app: about **4.4 GB compressed**
to download or copy, instead of about 8.9 GB, and about half the disk space. A
dependency-only change rebuilds only the last layers.

## Requirements

| | Needed |
|---|---|
| Device | NVIDIA Jetson Orin (Nano, NX, AGX) |
| JetPack | **6.1 – 6.2.x**: L4T R36.4 or a later R36 release, i.e. CUDA 12.6, cuDNN 9, TensorRT 10.3 |
| Host packages | the JetPack runtime (`sudo apt install nvidia-jetpack`), so that `/usr/local/cuda-12.6` and the TensorRT and cuDNN libraries exist |
| Docker | with the NVIDIA container runtime (`docker info` lists `nvidia` under Runtimes) |

Not supported: JetPack 5.x (L4T R35) and 6.0 (R36.3, TensorRT 8.6), JetPack 7, x86_64
machines. Use the root `Dockerfile` there.

Check a device first; nothing on it is changed:

```bash
bash docker/jetson-lean/host_check.sh
```

```text
✅ L4T R36.4.3
✅ TensorRT 10.3.0.30-1+cuda12.5
✅ CUDA 12.6
✅ libnvinfer.so.10
...
✅ Docker with the nvidia runtime
This host can run the Jetson lean image.
```

Each failed line says what is missing and how to fix it.

## Build

On the Jetson itself (the TensorRT version is read from the host):

```bash
git lfs pull
bash docker/jetson-lean/build.sh            # tags tinynav:jetson-lean
bash docker/jetson-lean/build.sh --no-app   # without the web app, smaller and faster
```

On an x86_64 machine through arm64 emulation (`qemu-user-static` / binfmt), naming the
TensorRT version of the Jetsons the image is for (`dpkg-query -W libnvinfer10` on one of
them). Expect hours: gtsam and several other packages compile under emulation.

```bash
bash docker/jetson-lean/build.sh --cross --trt-version 10.3.0.30-1+cuda12.5
```

Or plain Docker, from the repository root:

```bash
docker build -f docker/jetson-lean/Dockerfile \
  --build-arg TRT_VERSION=10.3.0.30-1+cuda12.5 -t tinynav:jetson-lean .
```

## Run

```bash
bash docker/jetson-lean/run.sh                       # a shell, this checkout at /tinynav
bash docker/jetson-lean/run.sh -- bash /tinynav/scripts/run_rosbag_examples.sh
bash docker/jetson-lean/run.sh --no-workspace        # the code baked into the image
```

`run.sh` checks the host, then starts the container with the same arguments as
`.devcontainer/devcontainer.json` (host network, `/dev`, X11, `~/.local/share/tinynav`),
the NVIDIA runtime, and these read-only mounts from the host:

- `/usr/local/cuda-12.6`;
- every `libnvinfer*`, `libnvonnxparser*` and `libcudnn*` library in
  `/usr/lib/aarch64-linux-gnu` (cuDNN 9 is split into several libraries, and TensorRT
  loads its builder resource when building engines), file by file, so the container
  keeps its own system libraries;
- `/usr/lib/aarch64-linux-gnu/nvidia`, only when the NVIDIA runtime does not provide it
  itself (newer runtimes do, and write into it).

To start it another way, mount the same paths: `host_check.sh --mounts` prints them as
`host:container:ro`. Without them the container stops at start and says what is
missing. `TINYNAV_SKIP_GPU_CHECK=1` starts it anyway, for CPU-only use and checks
(TensorRT cannot load then).

The VS Code dev container keeps using the root image: its run arguments cannot list
the host's libraries file by file.

## Verify on a device

Inside the container:

```bash
python -c "import tensorrt; print(tensorrt.__version__)"   # the host's version, e.g. 10.3.0
trtexec --version
make -C /tinynav/tinynav/models all                        # builds the TensorRT engines on this GPU
```

TensorRT engines (`*.plan`) are specific to the GPU and TensorRT version that built
them, exactly as with the root image.

## When the host changes

The image is tied to the host's JetPack. After upgrading JetPack on the device, run
`host_check.sh` again and rebuild the image. If the new JetPack has a different
TensorRT, CUDA or cuDNN series, this image does not apply to it until this directory is
updated for it.

## Keeping it in step with the root Dockerfile

`docker/jetson-lean/Dockerfile` repeats the root Dockerfile's steps. `check_sync.py`
fails when the root Dockerfile changes and this one does not follow, and CI
(`.github/workflows/jetson_lean_check.yml`) runs it on every change to either file.
The allowed differences are listed in `check_sync.py` and in the Dockerfile's header,
each with its reason.

```bash
python3 docker/jetson-lean/check_sync.py
python3 -m unittest discover -s docker/jetson-lean -p 'test_*.py'
```
