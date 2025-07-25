# Scripts Reference

This document provides an overview of the key scripts in the `scripts/` directory and how to use them for different TinyNav workflows.

# docker_run.sh

We provide pre-built [Docker images](https://hub.docker.com/r/uniflexai/tinynav/tags) to support both PC (x86_64) and NVIDIA Jetson (aarch64) environments. These images allow you to quickly launch the full TinyNav stack with all dependencies, whether you are developing on a desktop or deploying on an embedded platform.

- **PC (x86_64):** Use this image for standard desktop and laptop computers.
- **Jetson (aarch64):** Use this image for NVIDIA Jetson devices (e.g., Xavier, Nano) with ARM architecture.

Running `docker_run.sh` will launch the example pipeline.

# Development Inside Container

| Script                        | Description                                                                                   |
|-------------------------------|----------------------------------------------------------------------------------------------|
| `run_planning.sh`             | Runs the planning node for collision-free path generation (default planning mode).            |
| `run_navigation.sh`           | Runs the full navigation pipeline using a pre-built map for localization and planning.        |
| `run_rosbag_examples.sh`      | Runs a demo pipeline: launches core nodes, plays a sample rosbag, and opens RViz for visualization. |
| `run_rosbag_build_map.sh`     | Builds a map from a specified rosbag file, launching all required nodes and RViz.            |
| `run_realsense_sensor.sh`     | Starts the RealSense camera ROS 2 driver on your host system.                                |
| `run_realsense_bag_record.sh` | Records RealSense camera and IMU data into a rosbag for later mapping or playback.            |

TinyNav supports several key modes to fit different robotics workflows:

1. **Planning Only**
   - In this mode, TinyNav generates collision-free paths as you guide your robot using a joystick or keyboard.
   - To start planning-only mode, simply run:
     ```bash
     bash scripts/run_planning.sh
     ```

2. **Mapping Only**
   - Use this mode to build a map of a specific environment.
   - **Step 1: Record data from the RealSense camera**
     - On the machine connected to the RealSense camera, start the camera driver:
       ```bash
       bash scripts/run_realsense_sensor.sh
       ```
     - In a new terminal, record the camera and IMU data into a rosbag:
       ```bash
       bash scripts/run_realsense_bag_record.sh
       ```
     - When finished, stop both processes. The recorded rosbag file will be saved for mapping.
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
