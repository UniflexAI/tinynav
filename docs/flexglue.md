# Optional FlexGlue matching

TinyNav can use the FlexGlue v0.1 sparse TensorRT matcher in place of
LightGlue for map building and map relocalization. LightGlue remains the default;
real-time VIO in `perception_node` continues to use LightGlue.

SuperPoint keypoints, DINOv2/VLAD retrieval, depth filtering, PnP acceptance and
candidate ranking keep their existing behavior. The optional backend returns
matches in the original SuperPoint indexing and pixel-coordinate convention.

## Enable

Inside the usual TinyNav ROS/CUDA environment:

```bash
git lfs pull --include="tinynav/models/flexglue*.onnx"
make -C tinynav/models flexglue

python -m tinynav.core.build_map_node \
  --bag_file /path/to/bag --map_save_path /path/to/new_map \
  --feature_matcher flexglue

python -m tinynav.core.map_node \
  --tinynav_map_path /path/to/map --feature_matcher flexglue
```

Omit `--feature_matcher` or select `lightglue` for the existing implementation.
Python callers can pass `feature_matcher="flexglue"` to either node constructor.

Git LFS supplies the descriptor and TensorRT-compatible matcher ONNX build inputs.
The Makefile uses `trtexec` to build both architecture-suffixed `.plan` files.
Build on the target GPU with its installed TensorRT version. Both nodes load the
plans from `tinynav/models/`, or from `TINYNAV_FLEXGLUE_MODEL_DIR` when set.

## Maps and storage

Existing maps remain usable: FlexGlue reads their saved infrared images on demand
and keeps a 16-image dense-descriptor LRU (about 64 MiB). It does not rewrite their
feature stores. New keyframes built with FlexGlue include `flexglue_v0_1` in the
existing feature dictionary, with the dense descriptor and original image size.
This costs approximately 4 MiB per keyframe and permits matching while scratch
videos are still being written. Their SuperPoint features remain available, so
these maps can also be used with LightGlue. No pose, depth or ROS schema changes.

## Corridor replay on Orin Nano

The offline replay used Jetson Orin Nano Super 8GB at 15W, TensorRT 10.3 and CUDA
12.6. Both FlexGlue stages used device-built TensorRT engines. The same replay
also ran the prior ONNX Runtime CUDA implementation on the same inputs.

| Timing (ms) | LightGlue TensorRT | FlexGlue TensorRT | FlexGlue ONNX Runtime CUDA (prior) |
|---|---:|---:|---:|
| Single pair, median | 9.9 | 502.2 | 726.4 |
| Single pair, P95 | 10.7 | 504.8 | 728.1 |
| Three candidate matches, median | 29.8 | 1507.0 | 2179.5 |
| Measured stages with cached references, median | 52.8 | 1544.0 | 2308.8 |

Pair timing covers 192 matches with descriptors prepared. The last row includes
query descriptor preparation, matching, depth correspondence construction and PnP.
It excludes retrieval, disk I/O, ROS and queueing.

Across 64 requests from real corridor maps 3–10, both FlexGlue runtimes gave
31 map-consistent poses, 2 inconsistent poses and 31 requests without a pose.
LightGlue gave 32, 22 and 10 respectively. Consistency uses position error
below 0.75 m and rotation error below 10° against optimized map poses.
The replay used fixed Top-3 references and stored front-camera SuperPoint features.

## Upstream models

[FlexGlue](https://github.com/cvg/FlexGlue) is MIT-licensed. Its descriptor includes
DINOv3 weights subject to the separate
[DINOv3 license](https://github.com/cvg/FlexGlue/blob/main/DINOv3_LICENSE.md).
The model license notices are included in `tinynav/models/licenses/`.
