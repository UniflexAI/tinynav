<div align="center">

<picture>
  <img alt="tinynav logo" src="/docs/tinynav.png" width="50%" height="50%">
</picture>

**TinyNav** : *A lightweight, hackable system to guide your robots anywhere.*
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


# Bounties

We’ve launched our bounty program! Check the [list](https://docs.google.com/spreadsheets/d/1fyFSkiyfSGVeO8uW97gS7-gIt9qTbGIpYaMcHcjPF4Q/edit?usp=sharing) to see how you can contribute and the reward values for each task.

# Stereo Cameras

we’re excited to add [Looper](https://looper-robotics.com/?utm_source=blogger&utm_medium=social&utm_campaign=booking&promo=uIwjGxmJ) as a first-class supported camera, alongside RealSense.

Looper is special because it provides built-in depth and visual–inertial odometry (VIO), enabling many new possibilities for perception and navigation.

   <p align="center">
     <img alt="looper" src="/docs/looper.jpg" width="50%" height="50%">
   </p>


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
  - `stair_target.py` – Stair mode local target generator (experimental, see [Stair Mode](#stair-mode-experimental)).
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

#### Recommended: install a newer Node.js/npm with nvm first

Some systems ship with an older npm. We recommend installing a newer Node.js/npm via `nvm` before installing the Dev Containers CLI:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
nvm install --lts
```

Then install and use the Dev Containers CLI:

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

### Optional Dependencies

Depending on your robot platform or map representation, install the corresponding extras:

```bash
# Unitree GO2 robot support
uv sync --extra unitree

# LeKiwi robot support
uv sync --extra lekiwi

# 3D Gaussian Splatting (3DGS) map support
uv sync --extra 3dgs
```

You can combine multiple extras in one command:

```bash
uv sync --extra unitree --extra 3dgs
```

## Stair Mode (Experimental)

In stairwells map relocalization is unreliable (repetitive steps, one floor's map reused for every floor). Stair mode ignores the map and follows local geometry; it only needs to know whether to go `up` or `down`.

How it works (`tinynav/core/stair_target.py`):
1. Build a 2.5D height map around the robot from the last 3 s of depth (walls/railings are cells with a large z-span).
2. Search from the feet over cells whose height change fits a stair step.
3. Target = 1.2 m ahead on the path to the highest (`up`) or lowest (`down`) reachable cell.
4. On landings, where the next flight is not visible yet, explore toward the stairwell side (estimated online).
5. Odometry guard: every raw odometry pose (100 Hz) is fed in; if two consecutive poses are more than 10 cm apart the VIO is failing, so the height map is dropped and no target is given until 2 s without a jump.

Each call returns a `status`:

| Status | Meaning | Robot should |
|---|---|---|
| `ok` | A higher (`up`) / lower (`down`) level is reachable | Go to the target |
| `search` | Nothing higher/lower is reachable yet, usually a landing | Go to the exploration target (or turn in place) |
| `no_seed` | No ground found around the feet | Stop |
| `odom_invalid` | Odometry jumped within the last 2 s | Stop |

### Running stair mode

`tinynav/core/stair_node.py` replaces `map_node.py` as the source of `/control/target_pose` (run one or the other, not both). With the Looper camera:

```bash
bash scripts/run_looper_stair_navigation.sh
# in the last pane, pick the direction and press Enter
ros2 topic pub --once /stair/cmd std_msgs/msg/String '{data: down left}'   # or "up right", "down", "stop"
```

The optional second word is the U-turn side at landings (`left` / `right`, `auto` if omitted). `auto` works out the side while walking a flight: the next flight is behind the railing, not the wall, and a wall hides what is beyond it while a railing lets the camera see the parallel flight, so the side that shows more is taken. A stairwell always turns the same way, so giving the side is still the safest choice. The app's Stairs button asks for it too.

| Topic | Direction | Notes |
|---|---|---|
| `/slam/depth` + `/slam/odometry_visual` | in | same synced pair as `planning_node`, feeds the height map every frame |
| `/slam/odometry` | in | raw 100 Hz odometry for the jump check |
| `/stair/cmd` (`std_msgs/String`) | in | `up` / `down` (optionally `left` / `right`) start stair mode, `stop` leaves it |
| `/control/target_pose` | out | 2 Hz (`--rate`), same format as `map_node` |
| `/mapping/poi_change` | out | sent once when the robot must stop (`no_seed` / `odom_invalid`, or `stop`); planning drops its target and `cmd_vel_control` stops within ~0.8 s |
| `/stair/obstacles` | out | walls/railings seen in the last 5 s at the robot's height; in stair mode `planning_node` adds them to its obstacle map, because its free-space raycasting clears most of a railing through the gaps between bars. `--no_share_obstacles` turns this off |
| `/stair/status`, `/stair/path` | out | debugging: status string and the planned path |

`cmd_vel_control` only moves while `/nav/active` is true (normally set by the app). Set `--camera_height` to your robot's camera height above the ground.

### Stair memory (optional)

Geometry alone is weakest on landings. A stair memory remembers how people walked a given stairwell: each frame of a teleoperated recording is stored as its DINOv2 feature plus the direction the robot went over the next metre. At run time, if the camera image looks like a remembered place (cosine similarity >= 0.8), the remembered direction picks among the reachable, wall-clear targets; in unfamiliar places stair mode behaves as without a memory. Build it from recordings of the stairwell the robot will work in (a person driving the robot, up and down, a few times; Record bag in the app also keeps `/camera/camera/vio_100hz`):

```bash
# 1. image features, needs TensorRT (robot or dev container)
python tool/stair_memory.py features --bag <bag> --out <bag>_features.npz
# 2. label and merge, one --features/--direction per --bag
python tool/stair_memory.py build --bag A --features A_features.npz --direction down \
                                  --bag B --features B_features.npz --direction up --out stair_memory.npz
# run stair mode with it
python tinynav/core/stair_node.py --memory stair_memory.npz
# or evaluate offline with a bag that is not in the memory
.venv/bin/python tool/stair_offline_eval.py --bag C --direction down --stair-memory stair_memory.npz --features C_features.npz
```

Frames around odometry jumps, while backing up and while not making progress are left out of the memory. `/stair/status` shows the best similarity and `guided` when the memory picked the target.

### Replay test with planning

Replays a rosbag through `looper_bridge_node` + `planning_node` + `stair_node` in an isolated ROS domain (localhost only, so a robot on the network never sees these targets), records their outputs and renders a video:

```bash
bash scripts/run_stair_replay_test.sh tinynav_db/ros2bags/bag_downstairs down output/stair_replay left
```

`stair_replay.mp4` shows the camera image and planning's obstacle map with the planning trajectory (green), stair target (red), stair path (orange), recorded motion (white) and robot (yellow); STOP marks frames where stair_node told planning to stop. The replay is open loop: the robot follows the recording, so the video shows what planning would choose at each moment. The console prints how well the planning trajectory matches the recorded motion per phase.

### Offline evaluation

Evaluate the generator offline on a rosbag that contains depth, camera info and odometry:

```bash
source /opt/ros/humble/setup.bash
.venv/bin/python tool/stair_offline_eval.py \
    --bag tinynav_db/ros2bags/bag_downstairs \
    --direction down \
    --out output/stair_eval
```

| Option | Description (default) |
|---|---|
| `--direction` | `up` or `down` (required) |
| `--depth-topic` | `/camera/camera/depth/image_rect_raw` (`16UC1`/`mono16` in mm, or `32FC1` in m) |
| `--pose-topic` | `/camera/camera/vio_100hz` (`PoseStamped` or `Odometry`, camera pose in a z-up world frame) |
| `--image-topic` | `/camera/camera/infra1/image_rect_raw` (only for the video) |
| `--info-topic` | `/camera/camera/infra1/camera_info` |
| `--memory` | override the height map memory in seconds (default 3.0) |
| `--turn` | U-turn side at landings: `left`, `right` or `auto` (default) |
| `--rate` | target update rate in Hz, e.g. `2` to match `map_node` (default: every depth frame); odometry jumps still stop immediately |
| `--no-video` | skip writing the video |

Outputs in `--out`:
- `stair_eval.mp4`: camera image (left) and local height map (right).
- `stair_eval.csv`: per depth frame `t, phase, status, err_deg, cam_z, target_z`.
- Console summary: status counts, the time spans flagged `odom_invalid`, and the direction error between the target and the recorded motion 1 m ahead, split into `flight` / `landing` / `outside`.

Video legend (map is world-aligned, centered on the robot, 6 m x 6 m):

| Mark | Meaning |
|---|---|
| Yellow dot + arrow | Robot position and camera heading |
| Red dot | Target in `ok` state (a lower/higher level is reachable) |
| Amber dot | Target in `search` state (landing exploration or in-place turn) |
| Orange line | Planned path to the goal (the goal is its end) |
| White line | Recorded motion in the next 4 s (ground truth, not algorithm output) |
| Dark gray / light gray | Unobserved / wall or railing |
| Cyan to magenta | Height relative to the feet, cyan is lower |
| Green tint | Reachable from the feet |

Tunables live in `StairConfig` (e.g. `camera_height`, `max_step`, `robot_radius`, `jump_dist`, `jump_hold_s`); set `camera_height` to your robot's camera height above the ground.

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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/xiaolefang-dm"><img src="https://avatars.githubusercontent.com/u/62272320?v=4?s=100" width="100px;" alt="Xiaole Fang"/><br /><sub><b>Xiaole Fang</b></sub></a><br /><a href="https://github.com/UniflexAI/tinynav/commits?author=xiaolefang-dm" title="Code">💻</a></td>
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

**Looper Robotics** - https://looper-robotics.com/

