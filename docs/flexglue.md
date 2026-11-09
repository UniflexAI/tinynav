# Optional FlexGlue matching

TinyNav can use the official FlexGlue v0.1 sparse ONNX matcher in place of
LightGlue for map building and map relocalization. LightGlue remains the default;
real-time VIO in `perception_node` continues to use LightGlue.

SuperPoint keypoints, DINOv2/VLAD retrieval, depth filtering, PnP acceptance and
candidate ranking keep their existing behavior. The optional backend returns
matches in the original SuperPoint indexing and pixel-coordinate convention.

## Enable

Inside the usual TinyNav ROS/CUDA environment:

```bash
uv sync --extra flexglue
python -m tool.download_flexglue

python -m tinynav.core.build_map_node \
  --bag_file /path/to/bag --map_save_path /path/to/new_map \
  --feature_matcher flexglue

python -m tinynav.core.map_node \
  --tinynav_map_path /path/to/map --feature_matcher flexglue
```

Omit `--feature_matcher` or select `lightglue` for the existing implementation.
Python callers can pass `feature_matcher="flexglue"` to either node constructor.

The downloader verifies both official models by SHA256 and installs them under
`tinynav/models/flexglue/` (about 439 MiB). Downloads are explicit, not performed
while a node starts. For another directory, use `--model_dir` when downloading
and set `TINYNAV_FLEXGLUE_MODEL_DIR` when running.

A CUDA-enabled ONNX Runtime and its matching CUDA/cuDNN libraries are required.
The tested environment uses ONNX Runtime GPU 1.23.2 and cuDNN 9.10.2.21. Jetson
requires a JetPack-compatible ONNX Runtime CUDA build; a desktop PyPI wheel is
not a Jetson deployment. The backend fails if CUDA cannot initialize instead of
silently using a CPU-only session.

## Maps and storage

Existing maps remain usable: FlexGlue reads their saved infrared images on demand
and keeps a 16-image dense-descriptor LRU (about 64 MiB). It does not rewrite their
feature stores. New keyframes built with FlexGlue include `flexglue_v0_1` in the
existing feature dictionary, with the dense descriptor and original image size.
This costs approximately 4 MiB per keyframe and permits matching while scratch
videos are still being written. Their SuperPoint features remain available, so
these maps can also be used with LightGlue. No pose, depth or ROS schema changes.

## Corridor replay on Orin Nano

The optional FlexGlue backend runs both the descriptor and the sparse matcher with ONNX Runtime’s CUDA execution provider. The default LightGlue backend uses TensorRT.

On a Jetson Orin Nano Super 8GB at 15W, we replayed 64 front-camera requests from real corridor maps 3–10. Both backends received the same stored SuperPoint features, three candidate references per request and depth/PnP acceptance rules. Half the requests used overlapping maps and half used maps selected for distant look-alike frames.

| Nano replay timing (ms) | LightGlue · TensorRT | FlexGlue · ONNX Runtime CUDA |
|---|---:|---:|
| Single pair, median | 9.9 | 726.4 |
| Single pair, P95 | 10.7 | 728.1 |
| Three candidate matches per request, median | 29.8 | 2,179.5 |
| Timed stages with reference descriptors cached, median | 52.8 | 2,308.8 |

Single-pair statistics cover all 192 matcher calls with descriptors prepared. The last row sums query descriptor preparation, the three matcher calls, depth correspondence construction and PnP for each request, then takes the median across requests. SuperPoint extraction, retrieval, disk I/O, ROS and queueing are outside that measurement. These timings compare the implementations shipped by TinyNav on this device.

| Outcome across 64 requests | LightGlue · TensorRT | FlexGlue · ONNX Runtime CUDA |
|---|---:|---:|
| Pose consistent with map reference | 32 | 31 |
| Accepted pose inconsistent with map reference | 22 | 2 |
| No accepted pose | 10 | 31 |

Consistency requires position error below 0.75 m and rotation error below 10° against the optimized map pose. The deliberately selected, correlated samples measure behavior on this corridor. The replay used CUDA 12.6, TensorRT 10.3.0 and ONNX Runtime GPU 1.23.0.

## Implementation-specific corridor diagnostic

This experiment compares the existing LightGlue TensorRT FP16 implementation
with the official FlexGlue ONNX CUDA implementation. It is **not a controlled
comparison of model speed or intrinsic accuracy**: execution backends and
optimization paths differ. In particular, latency ratios cannot be attributed
to the matcher architectures. Both timing and pose outcomes below apply only to
these specific implementations and settings.

64 localization requests from front-camera maps 3-10: 32 requests against an
overlapping map and 32 against maps selected using distant similar frames.
Both implementations use the same images, SuperPoint keypoints, target-map VLAD
Top-3 candidates and existing `rerank_by_pnp_inliers` acceptance rules. FlexGlue
uses its own dense descriptor. The target map's vocabulary encodes query VLAD
from stored patch tokens.

| Outcome | LightGlue TensorRT | FlexGlue ONNX CUDA |
|---|---:|---:|
| Map-reference-consistent poses | 31 / 64 | 31 / 64 |
| Accepted map-reference mismatches | 24 / 64 | 2 / 64 |
| Requests with no accepted pose | 9 / 64 | 31 / 64 |

The lower mismatch count accompanies more requests without an accepted pose;
there is no increase in the total consistent-pose count. This does not establish
general accuracy superiority. Two LightGlue-consistent requests are rejected by
FlexGlue, while two other requests change from inconsistent to consistent.

Consistency means position error below 0.75 m and full rotation error below
10 degrees against saved query keyframe poses in the common map frame. These are
map-derived references, not surveyed ground truth. Saved query poses are used
only for evaluation. Samples are correlated and deliberately challenging, so
they do not estimate normal operating error rates.

| Implementation cost on shared RTX 3070 Laptop GPU | LightGlue TensorRT | FlexGlue ONNX CUDA |
|---|---:|---:|
| Median pair-matching time, cached descriptors | 3.3 ms | 122.3 ms |
| Median summed Top-3 computation | 41.6 ms | 433.5 ms |

These are deployment-implementation costs, **not an equal-backend speed
benchmark**. Reference descriptors are cached; FlexGlue's summed computation
includes one query DINOv3 description. The sum excludes live DINOv2 image
encoding, video decoding, ROS, queueing and planning, and is not measured
end-to-end application latency.

## Upstream models

[FlexGlue](https://github.com/cvg/FlexGlue) is MIT-licensed. Its descriptor includes
DINOv3 weights subject to the separate
[DINOv3 license](https://github.com/cvg/FlexGlue/blob/main/DINOv3_LICENSE.md).
Models are downloaded directly from the official v0.1 release; they are not
repackaged in this repository. The optional runtime requires no PyTorch install.
