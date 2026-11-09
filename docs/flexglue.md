# FlexGlue matching

Mapping, relocalization and native stereo VIO use FlexGlue directly. SuperPoint
keypoints, retrieval and PnP acceptance remain unchanged. Looper supplies its own odometry.

Inside the TinyNav ROS/CUDA environment, install the dependencies and fetch the weights once:

```bash
uv sync
python -m tool.download_flexglue
```

The downloader verifies both official models by SHA256 and installs them under
`tinynav/models/flexglue/` (about 439 MiB). Downloads are explicit, not performed
while a node starts. For another directory, use `--model_dir` when downloading
and set `TINYNAV_FLEXGLUE_MODEL_DIR` when running.

A CUDA-enabled ONNX Runtime and its matching CUDA/cuDNN libraries are required.
The tested environment uses ONNX Runtime GPU 1.23.2 and cuDNN 9.10.2.21. Jetson
requires a JetPack-compatible ONNX Runtime CUDA build; a desktop PyPI wheel is
not a Jetson deployment. The backend fails if CUDA cannot initialize instead of
silently using a CPU-only session. It does not load the experimental TensorRT
conversion from our separate performance investigation.

## Maps and storage

Existing maps remain usable: FlexGlue reads their saved infrared images on demand
and keeps a 16-image dense-descriptor LRU (about 64 MiB). It does not rewrite their
feature stores. New keyframes built with FlexGlue include `flexglue_v0_1` in the
existing feature dictionary, with the dense descriptor and original image size.
This costs approximately 4 MiB per keyframe and permits matching while scratch
videos are still being written. Their SuperPoint features remain available, so
these maps can also be used with LightGlue. No pose, depth or ROS schema changes.

## Corridor replay on Orin Nano

Measured 2026-11-07 on Jetson Orin Nano Super 8GB in its existing 15W mode.
This replays the **original real-corridor maps 3-10 and exact 64 requests** from
the desktop diagnostic: 32 overlapping-map requests and 32 requests against
maps selected using distant similar frames. It does not use the separate home
simulation bag. The 211 involved images, stored SuperPoint features, reference
depths, Top-3 candidate lists and PnP acceptance rules are fixed. Source map
pose/link/intrinsic hashes were checked against the original manifest.

Both the FlexGlue descriptor and matcher were converted to TensorRT on this
Nano. The descriptor uses an FP16-enabled build; the matcher uses the
FP16-preserving attention conversion and a strongly typed TensorRT build.
The accuracy replay uses the adapter from commit
`a9660347165a6a03364302f5f8152567e24e8bf7` on `xiaole/flexglue-matcher`, with
experimental TensorRT sessions substituted for its ONNX sessions.

### Localization outcomes

| Outcome / 64 requests | LightGlue TensorRT | FlexGlue TensorRT |
|---|---:|---:|
| Map-reference-consistent poses | 32 | 31 |
| Accepted map-reference mismatches | 22 | 2 |
| Requests with no accepted pose | 10 | 31 |

For these repetitive-corridor samples, FlexGlue substantially reduces wrong
accepted poses, which is useful when an incorrect map-to-odometry correction
can misdirect navigation. It gives up more requests instead: consistent poses
decrease by one and requests without a pose increase by 21. It still accepts
two inconsistent poses, so this is risk reduction, not elimination of wrong fixes.

Consistency means position error below 0.75 m and full rotation error below
10 degrees against optimized map poses. These references are not independent
surveyed truth. The 64 requests are correlated, deliberately difficult samples;
the counts do not estimate a robot's normal operating error rate.

**Conversion check:** FlexGlue TensorRT and Nano ONNX CUDA agree on the
accepted/rejected and consistent/inconsistent classification of all 64 requests.
Nano ONNX also agrees request-by-request with the original desktop FlexGlue
results (31 consistent, 2 inconsistent, 31 without a pose). This does not imply
bitwise-identical matches or floating-point outputs. Nano LightGlue differs from
the original desktop diagnostic on two requests, indices 0 and 21; the table
preserves the Nano measurements rather than substituting the desktop counts.

### Matched TensorRT timing

To control execution-wrapper differences, both matchers use the same synchronous
TensorRT host-to-host session wrapper, with CUDA Graph disabled for both.
24 image pairs are selected uniformly from the original 192 candidate pairs,
without selecting by either matcher's outcome. Each pair has two warmups and
three alternating measured repetitions: 72 measurements per matcher.
Descriptors are prepared before timing. LightGlue's matches from this wrapper
were checked against its existing CUDA Graph wrapper on the selected pairs.

| Pair-matching time | LightGlue TensorRT | FlexGlue TensorRT |
|---|---:|---:|
| Median | 12.7 ms | 501.8 ms |
| P95 | 13.6 ms | 504.2 ms |

These times include host/device transfers and matcher postprocessing. They
exclude descriptor extraction, SuperPoint, retrieval, PnP, disk, ROS, queueing
and planning. Three FlexGlue candidate matches alone take about 1.5 seconds
under these settings; this is not full-request latency. The descriptor was also
converted to TensorRT, but its execution is outside this matching-only table.
The main 64-request replay uses the existing LightGlue CUDA Graph wrapper for
its classifications; its timings are not substituted into the controlled table.

Environment: TensorRT 10.3.0, CUDA 12.6, ONNX Runtime GPU 1.23.0, OpenCV 4.11.0,
NumPy 1.26.1. Device-specific engines were built on the Nano rather than copied
from the desktop GPU.

**Integration status:** the TensorRT build/runner used here is an offline
experiment, not an enabled backend in this PR. The documented
`--feature_matcher flexglue` command still loads ONNX Runtime CUDA. This replay
does not validate live sensor scheduling, physical navigation, rear-camera
coverage or independent-capture generalization.

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

The Orin Nano section above supplies a separate matched-TensorRT timing test
and per-request conversion checks on this same dataset. Do not combine its
numbers with these mixed-backend desktop timings to infer a universal model
speed ratio. Independent-capture accuracy and live scheduling validation remain
outstanding. This PR does not change online localization acceptance thresholds.

## Upstream models

[FlexGlue](https://github.com/cvg/FlexGlue) is MIT-licensed. Its descriptor includes
DINOv3 weights subject to the separate
[DINOv3 license](https://github.com/cvg/FlexGlue/blob/main/DINOv3_LICENSE.md).
Models are downloaded directly from the official v0.1 release; they are not
repackaged in this repository. The optional runtime requires no PyTorch install.
