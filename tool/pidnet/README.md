# PIDNet floor segmentation tools

These scripts validate a PIDNet-S TensorRT floor segmentation engine before it
is wired into the runtime ROS graph.

The checked-in Cityscapes model treats `road` and `sidewalk` as the floor
candidate classes for early testing on `infra1`.

Recommended Jetson flow:

```bash
# Put the target-device TensorRT engine here:
/tinynav/tinynav/models/pidnet_s_cityscapes_256x320_aarch64.plan

python3 /tinynav/tool/pidnet/benchmark_pidnet_trt.py \
  --engine /tinynav/tinynav/models/pidnet_s_cityscapes_256x320_aarch64.plan

python3 /tinynav/tool/pidnet/pidnet_trt.py \
  --engine /tinynav/tinynav/models/pidnet_s_cityscapes_256x320_aarch64.plan \
  --image /path/to/image.png \
  --out /tmp/pidnet_floor_overlay.png \
  --mask-out /tmp/pidnet_floor_prob.png \
  --floor-channels 0,1 \
  --threshold 0.45
```

To publish live segmentation for other nodes:

```bash
python3 /tinynav/tool/pidnet/pidnet_segmentation_node.py \
  --engine /tinynav/tinynav/models/pidnet_s_cityscapes_256x320_aarch64.plan \
  --image-topic /camera/camera/infra1/image_rect_raw \
  --prob-topic /segmentation/floor_prob \
  --stable-prob-topic /segmentation/floor_prob_stable \
  --overlay-topic /segmentation/floor_overlay \
  --publish-hz 5.0 \
  --floor-channels 0,1 \
  --threshold 0.45 \
  --ema-current-weight 0.3 \
  --hysteresis-on 0.65 \
  --hysteresis-off 0.35 \
  --morph-open-kernel 3 \
  --morph-close-kernel 7
```

`/segmentation/floor_prob` is a `mono8` image where 0 means non-floor and 255
means floor candidate. `/segmentation/floor_overlay` is a `bgr8` visualization.
`/segmentation/floor_prob_stable` is the hysteresis + morphology stabilized
floor mask. At this stage, the segmentation topics are for visualization and
tuning only; planning fusion should be added later after the stable mask is
validated.

TensorRT `.plan` files are platform- and TensorRT-version-specific. Build them
on the Jetson or in a matching JetPack/TensorRT environment, then keep the
engine in `tinynav/models/` on the target machine. Do not commit generated
`.plan` files; commit the ONNX model and regenerate the engine per device.

If you have the ONNX file on the target device, build the engine with:

```bash
make -C /tinynav/tinynav/models pidnet
```

Override the workspace or build flags when needed:

```bash
make -C /tinynav/tinynav/models pidnet \
  PIDNET_WORKSPACE_FLAGS=--memPoolSize=workspace:256
```
