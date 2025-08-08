<div align="center">

<picture>
  <img alt="tinynav logo" src="/docs/tinynav.png" width="50%" height="50%">
</picture>

**TinyNav** /ˈtaɪni ˈnævi/: *A lightweight, hackable system to guide your robots anywhere.*
 Maintained by [Uniflex AI](https://x.com/UniflexAI).

<h3>

[Homepage](https://github.com/UniflexAI/tinynav) | [Documentation](./docs) | [Discord](https://discord.gg/gnZKFJ8W9Q)

</h3>

[![license](https://img.shields.io/github/license/UniflexAI/tinynav)](https://github.com/UniflexAI/tinynav/blob/master/LICENSE)
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FUniflexAI)](https://x.com/UniflexAI)
</div>

# Highlight (Our Design Goals)
We aim to make the system:

## Tiny
* Compact (~2000 LOC) for clarity and ease of use.
* Supports fast prototyping and creative applications.
* Encourages community participation and maintenance.

## Robust
* Designed to be reliable across diverse scenes and datasets.
* Ongoing testing for consistent performance in real-world conditions.

## Multiple Robots Platform
* Targeting out-of-the-box support for various robot types.
* Initial focus: [Lekiwi wheeled robot](https://github.com/SIGRobotics-UIUC/LeKiwi), [Unitree GO2](https://www.unitree.com/go2).
* Flexible architecture for future robot integration.

## Multiple Chips Platform
* Compute support starts with Jetson Orin and Desktop.
* Planning support for cost-effective platforms like RK3588.
* Aims for broader accessibility and deployment options.

# Project Structure

The repository is organized as follows:

- **`tinynav/core/`**  
  Core Python modules for perception, mapping, planning, and control:
  - `perception_node.py` – Processes sensor data for localization and perception.
  - `map_node.py` – Builds and maintains the environment map.
  - `planning_node.py` – Computes paths and trajectories using map and perception data.
  - `control_node.py` – Sends control commands to actuate the robot.
  - Supporting modules:
    - `driver_node.py`, `math_utils.py`, `models_trt.py`, `stereo_engine.py`.

- **`tinynav/cpp/`**  
  C++ backend components and bindings for performance-critical operations.

- **`tinynav/models/`**  
  Pretrained models and conversion scripts for perception and feature extraction.

- **`scripts/`**  
  Shell scripts for launching demos, managing Docker containers, and recording datasets.

---

# Getting Started

## Quick Start

```bash
git clone https://github.com/UniflexAI/tinynav.git
cd tinynav
bash scripts/docker_run.sh
```

You should see an RViz window displaying the live planning process.  
<p align="center">
  <img alt="tinynav logo" src="/docs/docker_run_rviz.png" width="50%" height="50%">
<p/>

---

## What `docker_run.sh` Does

The `docker_run.sh` script:

1. Checks the required environment (e.g., Docker with NVIDIA GPU support).
2. Launches the `run_rosbag_examples.sh` script, which:
   - Plays a recorded dataset from [this ROS bag](https://huggingface.co/datasets/UniflexAI/rosbag_d435i_01).
   - Runs the full TinyNav pipeline:
     - `perception_node.py`: Performs localization and builds the local map.
     - `planning_node.py`: Computes an optimal path for the robot.
     - RViz: Visualizes the robot's state and planned path in real time.

---

# Developer Guide

## Using Dev Containers

TinyNav supports [Dev Containers](https://containers.dev/) for a consistent and reproducible development experience.

### Using VS Code

1. Open the `tinynav` folder in Visual Studio Code.
2. Ensure the **Dev Containers** extension is installed.
3. VS Code will automatically start the container and open a terminal inside it.

### Using the Dev Container CLI

If you prefer the command line:

```bash
# Install the Dev Containers CLI
npm install -g @devcontainers/cli

# Start the Dev Container
devcontainer up --workspace-folder .

# Open a shell inside the container
devcontainer exec --workspace-folder . bash
```

---

## First-Time Setup (Inside the Dev Container)

After entering the development container, set up the Python environment:

```bash
uv venv --system-site-packages
uv sync
```

This will create a virtual environment and install all required dependencies.

## Set up for Unitree Go2

Connect your computer or Jetson to the Unitree Go2 and ensure they are on the same network segment (192.168.123.x).

### Step 1: Find the Network Interface Name

Run the following command to list all network interfaces:

```bash
ip addr
```

Example output:
```
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: wlP1p1s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 48:8f:4b:62:b4:4c brd ff:ff:ff:ff:ff:ff
    inet 192.168.19.115/24 brd 192.168.19.255 scope global dynamic noprefixroute wlP1p1s0
       valid_lft 12597sec preferred_lft 12597sec
    inet6 fe80::edfc:6e55:62ab:bfd3/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
3: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN group default qlen 10
    link/can 
4: eno1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 3c:6d:66:b2:b3:f1 brd ff:ff:ff:ff:ff:ff
    inet 192.168.123.100/24 scope global eno1
       valid_lft forever preferred_lft forever
```

**Note:** In this example, `eno1` is the network interface connected to the Go2 robot (IP: 192.168.123.100).

### Step 2: Set a Static IP Address

Configure your network interface with a static IP address in the 192.168.123.x range:

```bash
sudo ip addr add 192.168.123.100/24 dev eno1
sudo ip link set eno1 up
```

### Step 3: Verify Connection

Test the connection to the Go2 robot (default IP: 192.168.123.161):

```bash
ping 192.168.123.161
```

### Additional Resources

For more detailed setup instructions, refer to the official Unitree documentation:
https://support.unitree.com/home/en/developer/Quick_start

# Next Steps
- [ ] **LeKiwi integration**:
      Support closed-loop control with real-time processing.
- [ ] **Map module enhancement**:
      Enable map point editing, global navigation commands, and GUI-assisted optimization.
- [ ] **Embedded-friendly NN models**:
      Deliver lightweight neural network models for feature matching and stereo depth, optimized for platforms like RK3588.

# 📊 Line of Code
```
------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                          11            328            154           1959
C++                              3             49             32            292
Markdown                         2             76              6            167
Bourne Shell                     8              9              8            109
Dockerfile                       1             12             10             46
TOML                             1              6              0             33
JSON                             1              4              0             25
CMake                            1              4              0             16
XML                              1              0              0             13
-------------------------------------------------------------------------------
SUM:                            29            488            210           2660
-------------------------------------------------------------------------------
```


# Team

We are a small, dedicated team with experience working on various robots and headsets.



## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dvorak0"><img src="https://avatars.githubusercontent.com/u/2220369?v=4?s=100" width="100px;" alt="YANG Zhenfei"/><br /><sub><b>YANG Zhenfei</b></sub></a><br /><a href="https://github.com/UniflexAI/tinynav/commits?author=dvorak0" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/junlinp"><img src="https://avatars.githubusercontent.com/u/16746493?v=4?s=100" width="100px;" alt="junlinp"/><br /><sub><b>junlinp</b></sub></a><br /><a href="https://github.com/UniflexAI/tinynav/commits?author=junlinp" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/heyixuan-DM"><img src="https://avatars.githubusercontent.com/u/105199938?v=4?s=100" width="100px;" alt="heyixuan-DM"/><br /><sub><b>heyixuan-DM</b></sub></a><br /><a href="https://github.com/UniflexAI/tinynav/commits?author=heyixuan-DM" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/xinghanDM"><img src="https://avatars.githubusercontent.com/u/193573671?v=4?s=100" width="100px;" alt="xinghan li"/><br /><sub><b>xinghan li</b></sub></a><br /><a href="https://github.com/UniflexAI/tinynav/commits?author=xinghanDM" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!


## Sponsors ❤️
Thanks to our sponsor(s) for supporting the development of this project:

**DeepMirror** - https://www.deepmirror.com/

