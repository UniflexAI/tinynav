Welcome to the docs for tinyNav.


## Overview

**TinyNav** is a compact, efficient, and modular navigation framework designed primarily for use with [Intel RealSense depth cameras](https://www.intelrealsense.com/depth-camera-d435/) and NVIDIA [Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/product-development/) platforms. It provides essential capabilities for stereo-based visual odometry, mapping, and path planning in real-time robotics applications.
  
![data flow](/docs/rosgraph.svg)

## Module introduce

### Stereo Visual Odometry Module

The odometry module is expected to use pairs of images for relative pose estimation through feature extraction and feature matching procedures.

| Function                          | Description                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| **Depth Estimation**          | Computes dense depth maps from stereo disparity images using **[Semi-Global Block Matching](https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html) (SGBM)**, enabling 3D reconstruction and localization.                 |
| **Feature Extraction & Matching** | Detects keypoints with **[SuperPoint](https://github.com/rpautrat/SuperPoint)** and matches them using **[LightGlue](https://github.com/cvg/LightGlue)**.          |
| **Pose Estimation**               | Estimates relative camera pose via 3D-2D correspondences using **[PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)** methods.      |
| **Odometry Node Publishing(Optional)**           | Publishes estimated pose as `nav_msgs/Odometry` and broadcasts the corresponding TF. |

### Planning Module

The planning module is responsible for generating optimal and feasible control commands for the robot, utilizing information from local perception.

| Function                            | Description                                                                                                                                   |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Local Dynamic Mapping**           | Generates a local occupancy grid map from accumulated depth frames and robot poses in real-time.                                              |
| **Trajectory Simulation & Scoring** | Uses **[Dynamic Window Approach (DWA)](https://en.wikipedia.org/wiki/Dynamic_window_approach)** to simulate multiple candidate trajectories, scoring them based on goal alignment, obstacle avoidance. |
| **Optimal Command Selection**       | Selects the best trajectory and computes the corresponding velocity commands for execution.                                                   |
| **Planned Command Node Publishing(optional)**       | Publishes planned command as `geometry_msgs/Twist` and local map as `nav_msgs/OccupancyGrid`.|

### Mapping  Module

The mapping module is responsible for providing the robot’s global position along with static map information.

| Function                  | Description                                                                                                                                                       |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Global Feature Extraction** | Extracts global image descriptors using **[DINO V2 embeddings](https://huggingface.co/docs/transformers/model_doc/dinov2)** for scene recognition, loop detection, and global similarity assessment.                               |
| **Local Bundle Adjustment**   | Performs local optimization of camera poses and 3D landmark positions using a **[Bundle Adjustment](http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment)** solver while respecting relative pose constraints.             |
| **Loop Closure**              | Identifies potential loop closure candidates via **DINO feature similarity**, refines alignment through [pose graph optimization](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/slam/pose_graph_2d/pose_graph_2d.cc), and updates the map via BA.       |
| **Map Node Publishing (Optional)**      | Publishes the path as `nav_msgs/Path`.                                                                |
| **Global Planning (Optional)**      | Can be extended to include global path planning using cost maps and [**A\* method**](https://en.wikipedia.org/wiki/A*_search_algorithm).                                                                |


# Set up for Unitree Go2

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
