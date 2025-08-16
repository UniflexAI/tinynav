# FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 for amd64
# FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 for arm64
FROM uniflexai/base_image:latest
ENV ARCH=$ARCH
RUN echo "ARCH is $ARCH"

# Configure image
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
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

# Setup `python`
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install TensorRT based on architecture
RUN if [ "$ARCH" = "x86_64" ]; then \
        echo "Installing TensorRT 10.13.0.35-1+cuda12.9 for x86_64"; \
        apt-get update && apt-get install -y --no-install-recommends \
            libnvinfer10=10.13.0.35-1+cuda12.9 \
            libnvinfer-plugin10=10.13.0.35-1+cuda12.9 \
            libnvinfer-lean10=10.13.0.35-1+cuda12.9 \
            libnvinfer-vc-plugin10=10.13.0.35-1+cuda12.9 \
            libnvinfer-dispatch10=10.13.0.35-1+cuda12.9 \
            libnvonnxparsers10=10.13.0.35-1+cuda12.9 \
            libnvinfer-bin=10.13.0.35-1+cuda12.9 \
            libnvinfer-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-headers-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-plugin-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-headers-plugin-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-lean-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-vc-plugin-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-dispatch-dev=10.13.0.35-1+cuda12.9 \
            libnvonnxparsers-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-samples=10.13.0.35-1+cuda12.9 \
            libnvinfer-win-builder-resource10=10.13.0.35-1+cuda12.9 \
            python3-libnvinfer=10.13.0.35-1+cuda12.9 \
            python3-libnvinfer-dispatch=10.13.0.35-1+cuda12.9 \
            python3-libnvinfer-lean=10.13.0.35-1+cuda12.9 \
            python3-libnvinfer-dev=10.13.0.35-1+cuda12.9 \
            libnvinfer-headers-python-plugin-dev=10.13.0.35-1+cuda12.9 \
        && rm -rf /var/lib/apt/lists/*; \
    elif [ "$ARCH" = "aarch64" ]; then \
        echo "Installing TensorRT 10.13.2.6-1+cuda13.0 for aarch64"; \
        apt-get update && apt-get install -y --no-install-recommends \
            tensorrt=10.3.0.30-1+cuda12.5 \
        && rm -rf /var/lib/apt/lists/*; \
    else \
        echo "Unsupported architecture: $arch"; exit 1; \
    fi

# ros2
RUN apt-get update && apt-get install -y software-properties-common \
    && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository universe
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && apt-get install -y ros-humble-desktop \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# env
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
ENV CYCLONEDDS_URI=/tinynav/scripts/cyclone_dds_localhost.xml
ENV PATH=$PATH:/usr/src/tensorrt/bin/

# install cyclonedds for unitree sdk
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
RUN cd cyclonedds && mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=../install && make -j$(nproc) && make install
ENV CYCLONEDDS_HOME="/cyclonedds/install"

# gazebo
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN apt-get update
RUN apt-get install -y ignition-fortress ros-humble-ros-gz-bridge


WORKDIR /3rdparty

# realsense sdk
RUN apt-get update && apt-get install -y --no-install-recommends \
    libudev-dev pkg-config libusb-1.0-0-dev libglfw3-dev libssl-dev libgl1-mesa-dev libglu1-mesa cmake \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/IntelRealSense/librealsense.git -b r/256 \
    && cd librealsense \
    && cp config/99-realsense-libusb.rules /etc/udev/rules.d/. \
    && mkdir build \
    && cd build \
    && cmake ../ -DFORCE_LIBUVC=true -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true \
    && make -j$(nproc) \
    && make install
    
# realsense ros2 wrapper
ENV ROS_DISTRO=humble
RUN mkdir -p ros2_ws/src \
    && cd ros2_ws/src \
    && git clone https://github.com/UniflexAI/realsense-ros.git -b ros2-master \
    && cd /3rdparty/ros2_ws \
    && apt-get update && apt-get install -y python3-rosdep \
    && rosdep init \
    && rosdep update \
    && rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y && rm -rf /var/lib/apt/lists/* \
    && . /opt/ros/humble/setup.sh \
    && colcon build

RUN echo "source /3rdparty/ros2_ws/install/local_setup.bash" >> ~/.bashrc

# clean
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# the project
WORKDIR /tinynav
USER root

# Install uv
RUN curl -LsSf https://astral.sh/uv/0.7.3/install.sh | sh
ENV PATH=$PATH:/root/.local/bin/

COPY ./tinynav        /tinynav/tinynav/
COPY ./scripts        /tinynav/scripts/
COPY ./docs           /tinynav/docs/
COPY ./pyproject.toml /tinynav/
COPY ./uv.lock        /tinynav/
RUN chmod +x /tinynav/scripts/*.sh

RUN uv venv --system-site-packages
RUN uv sync
