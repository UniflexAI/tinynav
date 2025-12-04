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

| [Unitree GO2](https://www.unitree.com/go2)  | [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) |
| ------------- | ------------- |
| <video src="https://github.com/user-attachments/assets/f4ff4842-f0ca-4299-b3d5-5d097de1f2ba">  | <video src="https://github.com/user-attachments/assets/c9b4b949-943b-4910-92f0-0337ef26d0b0">   |

| [Navigation with 3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) |
| ----------------------|
| <video src="https://github.com/user-attachments/assets/5e0d5846-ab3f-4a57-8bdd-067473e758a9"> |

| Vision Only Mapping |
| ----------------------|
| <video src="https://github.com/user-attachments/assets/578baeb2-63f7-444c-a9cd-e6a8036151d3"> |



# [v0.3] What's Changed
## 🚀 Features
- **IMU–Visual Fusion in Perception Node**
  
    Integrates IMU–visual fusion to significantly improve pitch-angle accuracy.
  
    This enhancement boosts overall robustness and enables reliable navigation across more robot platforms, especially those sensitive to pitch drift.

- **Resilient Mapping Pipeline**

    Upgraded map-building logic to gracefully handle message loss, improving stability in real-world communication conditions.
  
    Paired with a redesigned visualization module, developers can now observe the map-building process incrementally, making debugging and tuning far more intuitive.

- **Unified Model Training for Perception + Planning**
  
    We have begun training a single neural model that jointly supports both perception and planning tasks, paving the way for tighter integration and future performance gains. [(TinyBEV)](https://github.com/uniflexai/tinybev)

## 🔧 Improvements
- **Enhanced C++ CI & Code Quality**
  
    The CI pipeline now includes:
    * clang-tidy static analysis
    * ASAN (Address Sanitizer) detection
      
    These additions ensure higher reliability, cleaner code, and safer memory usage across the C++ stack.

## 🐞 Bug Fixes

Dozens of internal fixes and refinements were merged this cycle, improving system stability, consistency, and developer experience.

# [v0.2] What's Changed
## 🚀 Features
- **3D Gaussian Splatting (3DGS) Map Representation**  
  Provides high-quality visualization and an intuitive map editor, making it easy to inspect map details and place target POIs with precision.

- **ESDF-based Obstacle Avoidance**  
  Enables more human-like navigation. Robots not only avoid obstacles but also keep a safe distance, improving path quality.

- **Localization Benchmark**  
  Adds a benchmark for map-based localization, allowing clear and quantitative evaluation of improvements across versions.

- **CUDA Graph Optimization**  
  Reduces inference overhead and achieves >20Hz on Jetson Nano, lowering latency for real-time closed-loop navigation.

## 🔧 Improvements
- **Simplified First-Time Setup**  
  The `postStartCommand` command in the dev container now auto-generates platform-specific models, reducing errors and making setup more user-friendly.

- **Expanded CI Testing**  
  Broader continuous integration coverage ensures higher build stability and code quality.

- **Map Storage with KV Database**  
  Maps are now stored using `shelve`, resulting in shorter code and better performance.

## 🐞 Bug Fixes
- Over **50 pull requests** merged since the last release, delivering numerous fixes and stability improvements.


# [v0.1] What's Changed
## 🚀 Features
* Implemented **map-based navigation** with **relocalization** and **global planning**.
* Added support for **Unitree** robots.
* Added support for the **Lewiki** platform.
* **Upgraded stereo depth model** for a better speed–accuracy balance.
* **Tuned Intel® RealSense™ exposure strategy**, optimized for robotics tasks.
* Added **Gazebo** simulation environment
* CI: **Docker image build & push** pipeline.
## 🔧 Improvements
* Used **Numba JIT** to speed up key operations while keeping the code simple and maintainable.
* Adopted **asyncio** for **concurrent model inference.**
* Added **gravity correction** when velocity is zero.
* Mount **/etc/localtime** by default so **ROS bag** files use local time in their names.
* **Optimized trajectory generation.**
## 🐞 BugFix
* Various bug fixes and stability improvements.

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
## Prerequisites

Before you begin, make sure you have the following installed:

- **git** and **git-lfs** (for cloning and handling large files)
- **Docker**

**Platform-specific requirements:**
- For **x86_64** (PC): [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support)
- For **Jetson Orin**: [JetPack SDK](https://developer.nvidia.com/embedded/jetpack) version 6.2 or higher

## 🚀 Quick Start

1. **Check the environment**
   ```bash
   git clone https://github.com/UniflexAI/tinynav.git
   cd tinynav
   bash scripts/check_env.sh
   ```
   Follow the instructions to fix any environment issues until you see:
   ```bash
   ✅ Docker is installed.
   ✅ Docker daemon is running and accessible.
   ✅ NVIDIA runtime is available in Docker.
   ✅ Git LFS is installed.
   ✅ devcontainer.json patched for your x86 platform.
   ```

2. **Open the project in VS Code**  
   - Launch VS Code and open the `tinynav` folder.  
   - Install the **Dev Containers** extension if prompted.  
   - Reopen the folder inside the container.  

3. **Run the example**  
   Once inside the container, start the demo:
   ```bash
   bash /tinynav/scripts/run_rosbag_examples.sh
   ```
   You should see an **RViz** window displaying the live planning process:  

   <p align="center">
     <img alt="tinynav logo" src="/docs/docker_run_rviz.png" width="50%" height="50%">
   </p>

---

## 📜 What `run_rosbag_examples.sh` Does

The script automates the entire demo workflow:

1. **Plays dataset**  
   Streams a recorded dataset from [this ROS bag](https://huggingface.co/datasets/UniflexAI/rosbag2_go2_exposure_1k).

2. **Runs TinyNav pipeline**  
   - **`perception_node.py`** → Performs localization and builds the local map.  
   - **`planning_node.py`** → Computes the robot’s optimal path.  
   - **RViz** → Visualizes the robot’s state and planned trajectory in real time.  

---

✨ With these steps, you’ll have the full TinyNav system up and running in minutes.
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

# Next Steps
- [ ] **High Optimization NN models**:
      Support real-time perception processing at >= 30fps.
- [ ] **Map module enhancement**:
      Improve consistency and accuracy for mapping and localization.
- [ ] **End-To-End trajectories planning**:
      Deliver robust and safe trajectories with integrated semantic information.

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

