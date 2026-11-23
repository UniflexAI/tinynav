# TinyNav for Jetson, reusing the host's JetPack ("lean" image).
#
# The root Dockerfile builds on a JetPack base image that carries CUDA, cuDNN and the
# TensorRT libraries inside the image, about 5 GiB of its ~8.9 GB compressed size.
# A Jetson running JetPack already has the very same libraries on the host. This image
# leaves them out and expects them to be mounted read-only at run time (run.sh does
# it); only the thin TensorRT Python binding and trtexec are added, pinned to the
# host's exact TensorRT version.
#
# Every other step is the root Dockerfile's, in its order; check_sync.py (CI) fails
# when the root Dockerfile changes and this one does not follow. Differences, each
# for a reason given where it happens:
#   * base ros:humble instead of the JetPack base (ROS's apt source is already in it);
#   * TensorRT binding + trtexec only, no TensorRT/CUDA/cuDNN libraries;
#   * rosdep is already initialised in ros:humble; /etc/udev/rules.d is created;
#   * the entrypoint is a file next to this one (works with the legacy builder too)
#     and first checks that the host's GPU libraries are mounted;
#   * the web app (Flutter) is optional: --build-arg WITH_APP=0 leaves it out.
#
# Supported hosts: Jetson Orin with JetPack 6.1 - 6.2.x (L4T R36.4+), i.e. CUDA 12.6,
# cuDNN 9 and TensorRT 10.3.0.30. See docker/jetson-lean/README.md.
#
# Build from the repository root (build.sh fills the versions from the host):
#   docker build -f docker/jetson-lean/Dockerfile -t tinynav:jetson-lean .
ARG BASE=ros:humble
FROM ${BASE}

ENV ARCH=aarch64
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    build-essential cmake \
    git git-lfs openssh-client \
    nano vim less util-linux tree \
    htop atop nvtop \
    sed gawk grep curl wget zip unzip \
    tcpdump sysstat screen tmux \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa \
    speech-dispatcher portaudio19-dev libgeos-dev \
    libceres-dev \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

# TensorRT: binding and trtexec only, unpacked without their library dependencies.
ARG TRT_VERSION=10.3.0.30-1+cuda12.5
ARG JETSON_REPO=https://repo.download.nvidia.com/jetson
RUN apt-get update && apt-get install -y --no-install-recommends gnupg ca-certificates \
    && curl -fsSL ${JETSON_REPO}/jetson-ota-public.asc | gpg --dearmor -o /usr/share/keyrings/jetson.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/jetson.gpg] ${JETSON_REPO}/common r36.4 main" \
       > /etc/apt/sources.list.d/jetson-common.list \
    && apt-get update && cd /tmp \
    && apt-get download python3-libnvinfer=${TRT_VERSION} libnvinfer-bin=${TRT_VERSION} \
    && dpkg -x python3-libnvinfer_*.deb / && dpkg -x libnvinfer-bin_*.deb / \
    && rm -f /tmp/*.deb /etc/apt/sources.list.d/jetson-common.list \
    && rm -rf /var/lib/apt/lists/*

# ros2 (ros:humble already has the archive key and source)
RUN apt-get update && apt-get install -y software-properties-common \
    && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository universe
RUN apt-get update && apt-get install -y ros-humble-desktop \
    ros-humble-image-transport-plugins \
    ros-humble-compressed-image-transport \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

ENV RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
ENV FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA
ENV OPENBLAS_NUM_THREADS=1
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
ENV CYCLONEDDS_URI=/tinynav/scripts/cyclone_dds_localhost.xml
ENV PATH=$PATH:/usr/src/tensorrt/bin/

RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
RUN cd cyclonedds && mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=../install && make -j$(nproc) && make install
ENV CYCLONEDDS_HOME="/cyclonedds/install"

RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN apt-get update
RUN apt-get install -y ignition-fortress ros-humble-ros-gz-bridge

WORKDIR /3rdparty
RUN apt-get update && apt-get install -y --no-install-recommends \
    libudev-dev pkg-config libusb-1.0-0-dev libglfw3-dev libssl-dev libgl1-mesa-dev libglu1-mesa cmake \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*
# realsense sdk r/256; ros:humble has no /etc/udev/rules.d yet
RUN git clone https://github.com/IntelRealSense/librealsense.git -b r/256 \
    && cd librealsense \
    && mkdir -p /etc/udev/rules.d && cp config/99-realsense-libusb.rules /etc/udev/rules.d/. \
    && mkdir build \
    && cd build \
    && cmake ../ -DFORCE_LIBUVC=true -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DCHECK_FOR_UPDATES=OFF \
    && make -j$(nproc) \
    && make install
# realsense ros2 wrapper; rosdep is already initialised in ros:humble
ENV ROS_DISTRO=humble
RUN mkdir -p ros2_ws/src \
    && cd ros2_ws/src \
    && git clone https://github.com/UniflexAI/realsense-ros.git -b ros2-master \
    && cd /3rdparty/ros2_ws \
    && apt-get update && apt-get install -y python3-rosdep \
    && (rosdep init || true) \
    && rosdep update \
    && rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y && rm -rf /var/lib/apt/lists/* \
    && . /opt/ros/humble/setup.sh \
    && colcon build
RUN echo "source /3rdparty/ros2_ws/install/local_setup.bash" >> ~/.bashrc
RUN echo 'setw -g mouse on' > /root/.tmux.conf

RUN apt-get update && \
    apt-get install -y clang-tidy ros-humble-ament-clang-tidy ros-humble-ament-lint \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install pyparsing==3.1.1
ARG GTSAM_REF=yzf/add_smart_factor_python_export
RUN git clone https://github.com/dvorak0/gtsam.git -b ${GTSAM_REF} \
    && cd gtsam \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DGTSAM_BUILD_PYTHON=ON -DGTSAM_THROW_CHEIRALITY_EXCEPTION=OFF .. \
    && make -j2
ENV PYTHONPATH="/3rdparty/gtsam/build/python:${PYTHONPATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    nlohmann-json3-dev \
    ros-humble-plotjuggler \
    ros-humble-plotjuggler-ros \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /3rdparty/plotjuggler_ws/src \
    && git clone https://github.com/dvorak0/plotjuggler-ros-plugins.git /3rdparty/plotjuggler_ws/src/plotjuggler_ros \
    && cd /3rdparty/plotjuggler_ws \
    && . /opt/ros/humble/setup.sh \
    && colcon build --packages-select plotjuggler_ros --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
RUN echo "source /3rdparty/plotjuggler_ws/install/local_setup.bash" >> ~/.bashrc

RUN apt-get update && apt-get install -y \
    ros-humble-foxglove-bridge \
    ros-humble-foxglove-msgs \
    ros-humble-foxglove-compressed-video-transport \
    && rm -rf /var/lib/apt/lists/*

# --allow-overriding comes from colcon-override-check, which the root Dockerfile's
# JetPack base gets with its ROS install and ros:humble does not have.
RUN apt-get update && apt-get install -y python3-colcon-override-check \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /3rdparty/message_filters_ws/src \
    && git clone --branch humble https://github.com/UniflexAI/message_filters.git /3rdparty/message_filters_ws/src/message_filters \
    && cd /3rdparty/message_filters_ws \
    && . /opt/ros/humble/setup.sh \
    && colcon build --packages-select message_filters --allow-overriding message_filters
RUN echo "source /3rdparty/message_filters_ws/install/local_setup.bash" >> ~/.bashrc

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /tinynav
USER root
RUN curl -LsSf https://astral.sh/uv/0.7.3/install.sh | sh && test -f /root/.local/bin/uv
ENV PATH=$PATH:/root/.local/bin/
ENV UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=8

# the project
# Pre-create venv and sync dependencies at build time
WORKDIR /tinynav
COPY . /tinynav
RUN rm -rf .venv
RUN /root/.local/bin/uv venv /opt/venv --system-site-packages --seed
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/tinynav:/3rdparty/gtsam/build/python:/opt/venv/lib/python3.10/site-packages"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# lekiwi conflicts with unitree (lerobot vs unitree-sdk2py dependency clash), so only unitree is included by default.
# To use lekiwi instead, replace --extra unitree with --extra lekiwi.
RUN /root/.local/bin/uv sync --python /opt/venv/bin/python --extra unitree

# build decord from source
RUN git clone --recursive https://github.com/dmlc/decord.git /tmp/decord \
    && cd /tmp/decord \
    && mkdir -p build \
    && cd build \
    && cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release \
    && make -j"$(nproc)" \
    && cd ../python \
    && /usr/bin/python3 setup.py install \
    && cp -f /tmp/decord/build/libdecord.so /usr/local/lib/libdecord.so \
    && DECORD_PKG_DIR=$(/usr/bin/python3 -c "import site; print(site.getsitepackages()[0] + '/decord')") \
    && ln -sf /usr/local/lib/libdecord.so "${DECORD_PKG_DIR}/libdecord.so" \
    && rm -rf /tmp/decord

# The root Dockerfile's entrypoint (model build prompt), behind a check that the
# host's GPU libraries are mounted.
COPY docker/jetson-lean/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Flutter — install stable via git (arch-agnostic: works on x86_64 and aarch64)
# Optional here: WITH_APP=0 leaves out the web app and its toolchain.
ARG WITH_APP=1
ENV PATH="/opt/flutter/bin:$PATH"
RUN if [ "$WITH_APP" = 1 ]; then \
        git clone https://github.com/flutter/flutter.git -b stable --depth 1 /opt/flutter \
        && git config --global --add safe.directory /opt/flutter \
        && flutter config --enable-web \
        && flutter precache --web \
        && cd /tinynav/app/frontend \
        && flutter pub get \
        && flutter build web --release --suppress-analytics; \
    fi

# What the host supplies at run time (run.sh mounts it read-only): CUDA under
# /usr/local/cuda-<version>, the TensorRT and cuDNN libraries in /usr/lib/aarch64-linux-gnu,
# and the driver through the NVIDIA container runtime.
ARG CUDA_VERSION=12.6
ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
ENV PATH=$PATH:/usr/local/cuda-${CUDA_VERSION}/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/nvidia
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
ENV TINYNAV_HOST_GPU="cuda-${CUDA_VERSION} cudnn-9 tensorrt-${TRT_VERSION}"
LABEL org.tinynav.host-gpu="cuda-${CUDA_VERSION} cudnn-9 tensorrt-${TRT_VERSION}"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
