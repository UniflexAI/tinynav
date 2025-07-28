#FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

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

# ros2
RUN apt-get update && apt-get install -y software-properties-common \
    && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository universe
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && apt-get install -y ros-humble-desktop \
    python3-colcon-common-extensions \
    ros-humble-librealsense2* \
    ros-humble-realsense2-* \
    && rm -rf /var/lib/apt/lists/*

# tensorrt
RUN apt-get update && apt-get install -y python3-libnvinfer \
    tensorrt \
    && rm -rf /var/lib/apt/lists/*

# clean
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# for root user
USER root

# Install uv
RUN curl -LsSf https://astral.sh/uv/0.7.3/install.sh | sh
ENV PATH=$PATH:/root/.local/bin/

# env
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
ENV CYCLONEDDS_URI=/tinynav/scripts/cyclone_dds_localhost.xml
ENV PATH=$PATH:/usr/src/tensorrt/bin/

# install cyclonedds for unitree sdk
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
RUN cd cyclonedds && mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=../install && make -j$(nproc) && make install
ENV CYCLONEDDS_HOME="/cyclonedds/install"

WORKDIR /tinynav

COPY ./tinynav        /tinynav/tinynav/
COPY ./scripts        /tinynav/scripts/
COPY ./docs           /tinynav/docs/
COPY ./pyproject.toml /tinynav/
COPY ./uv.lock        /tinynav/
RUN chmod +x /tinynav/scripts/*.sh

RUN uv venv --system-site-packages
RUN uv sync
