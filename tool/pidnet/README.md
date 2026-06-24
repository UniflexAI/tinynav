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
