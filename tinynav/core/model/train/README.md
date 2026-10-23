# BEV traversability model

Learn a BEV map of the ground around the robot from what is available at runtime (infra1 image, camera depth,
camera tilt), supervised by labels built offline from much stronger "teacher" models and many frames.
The network predicts robot-agnostic geometry; `cost.py` turns it into free / obstacle per robot type
(`quadruped` treats stairs as traversable, `wheeled` blocks on curbs and steps).

BEV grid (`bev_geometry.py`): 128 x 128 cells of 5 cm, x forward [-1.6, 4.8) m, y left [-3.2, 3.2) m,
gravity aligned, following the camera heading.

## 1. Labels (offline, GPU)

Teachers: [FoundationStereo](https://github.com/NVlabs/FoundationStereo) for depth (set `FOUNDATION_STEREO_DIR`,
weights in `pretrained_models/23-51-11`) and Mask2Former (ADE20K) for railings. Bags need
`infra1`, `infra2`, `depth` and VIO pose (`/camera/camera/vio_100hz` or `vio_image`).

```bash
PY=.venv/bin/python bash tinynav/core/model/train/labels/batch_labels.sh <rosbags_dir> <data_dir> bag_a:2 bag_b:1
```

Per bag this runs:
- `extract_frames` - synced left/right/depth at 1-2 Hz plus the full-rate pose track
- `run_teachers` - teacher depth (`*_fsdepth.npy`) and segmentation (`*_seg.png`), ~3 s/frame on an RTX 3060
- `make_bev_gt` - per frame, teacher points of the frames within +-30 s are fused in that frame's BEV;
  voxels seen through by more frames than saw them (people, cars, noise) are dropped. Layers:
  `known, ground, span, step, rail, traversed, visible` in `bev/{frame}.npz`

Checks: `viz_bev` (image | planning rule on camera depth | label | ground height) and `eval_baseline`
(how much of the label the current planning rule misses; traversed cells labelled obstacle = label noise).

## 2. Training

```bash
python -m tinynav.core.model.train.train --data <data_dir> --out runs/bev_v1
```

`BevNet` (`model.py`): ResNet-18 image features are sampled at 8 heights into the BEV (`grid_sample` with
grids precomputed from intrinsics and camera tilt), concatenated with a camera-depth raster, and decoded by a
small 2D U-Net into `known, ground, span, step, rail`. Validation reports obstacle recall / precision / IoU per
robot preset, split into cells seen by the current frame and cells filled in from other frames.

## 3. On the robot

`planning_node` can add BevNet obstacles to its obstacle map; it is off unless `TINYNAV_BEV_ENGINE` is set.

```bash
python -m tinynav.core.model.train.export_onnx --ckpt runs/<run>/best.pt --out bevnet.onnx
trtexec --onnx=bevnet.onnx --fp16 --saveEngine=bevnet_fp16_$(uname -m).plan   # on the target machine
TINYNAV_BEV_ENGINE=$PWD/bevnet_fp16_$(uname -m).plan python tinynav/core/planning_node.py
```

It needs `/camera/camera/infra1/image_rect_raw` besides the usual planning inputs (images are matched to
`/slam/depth` by stamp) and publishes the added cells on `/planning/bev_obstacle_mask`. `bev_runtime.py` holds
the shared logic: model obstacles are only trusted inside the camera view, never under the robot (0.35 m),
and need P >= 0.7. `replay_planning.py` runs the same code offline against recorded frames and writes videos.
