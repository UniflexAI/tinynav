# Scripts Reference

This document provides an overview of the key scripts in the `scripts/` directory and how to use them for different TinyNav workflows.

# Development Inside Container

| Script                        | Description                                                                                   |
|-------------------------------|----------------------------------------------------------------------------------------------|
| `run_planning.sh`             | Runs the planning node for collision-free path generation (default planning mode).            |
| `run_navigation.sh`           | Runs the full navigation pipeline using a pre-built map for localization and planning.        |
| `run_rosbag_examples.sh`      | Runs a demo pipeline: launches core nodes, plays a sample rosbag, and opens RViz for visualization. |
| `run_rosbag_build_map.sh`     | Builds a map from a specified rosbag file, launching all required nodes and RViz.            |
| `run_realsense_sensor.sh`     | Starts the RealSense camera ROS 2 driver on your host system.                                |
| `run_rosbag_record.sh`        | Records the mapping rosbag topics into the XDG data directory.                               |
| `run_map_record.sh`           | Launches RealSense in tmux, waits 3s, then calls the rosbag record helper.                   |

TinyNav supports several key modes to fit different robotics workflows:

1. **Planning Only**
   - In this mode, TinyNav generates collision-free paths while the robot is teleoperated (e.g., using a joystick or keyboard), as illustrated with the [LeKiwi mobile robot](https://huggingface.co/docs/lerobot/lekiwi).
   - To start planning-only mode, simply run:
     ```bash
     bash scripts/run_planning.sh
     ```

2. **Mapping Only**
   - Use this mode to build a map of a specific environment.
   - **Step 1: Record data from the RealSense camera**
     - On the machine connected to the RealSense camera, launch the combined recording workflow:
       ```bash
       bash scripts/run_map_record.sh
       ```
     - This script starts the RealSense driver in tmux, waits 3 seconds, then calls `run_rosbag_record.sh` to record the camera and IMU data into a rosbag.
     - Recorded bags are saved under `${XDG_DATA_HOME:-$HOME/.local/share}/tinynav/rosbags/` for later mapping.
   - **Step 2: Build the map from the recorded data**
     - Use the recorded rosbag to build a map:
       ```bash
       bash scripts/run_rosbag_build_map.sh
       ```
     - All generated maps will be saved in the `tinynav_map` directory.

   - You can also build a map using the included example data:
      ```bash
      bash scripts/run_rosbag_build_map.sh
      ```
    <picture>
      <img alt="prebuild-map" src="/docs/map.png" width="50%" height="50%">
    </picture>

3. **Map-Based Navigation**
   - Navigate your robot using a pre-built map created with the mapping workflow above.
   - In the map-based GUI, you can set up paths or points of interest (POIs) for the robot to follow.
   - To launch map-based navigation, run:
     ```bash
     bash scripts/run_navigation.sh
     ```
   - The robot will follow the designated POIs in the order you specify.
