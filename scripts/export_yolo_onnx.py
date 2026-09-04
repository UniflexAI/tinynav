"""One-time export of a pretrained COCO YOLO checkpoint to ONNX.

Not a runtime dependency of tinynav.core.models_trt.YoloDetectorTRT, which only
needs the compiled TensorRT engine. Run this once (on a machine with network
access and the `yolo_export` extra installed: `uv sync --extra yolo_export`),
then build the engine with `make -C tinynav/models yolo`, and commit both the
.onnx and the resulting .plan the same way dinov2/superpoint/etc are checked in.

Usage:
    python scripts/export_yolo_onnx.py --checkpoint yolo11n.pt --imgsz 640 \
        --output tinynav/models/yolo11n_640x640.onnx
"""
import tyro


def main(
    checkpoint: str = "yolo11n.pt",
    imgsz: int = 640,
    opset: int = 17,
    output: str = "tinynav/models/yolo11n_640x640.onnx",
) -> None:
    from pathlib import Path

    from ultralytics import YOLO

    model = YOLO(checkpoint)
    exported_path = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(exported_path).replace(output_path)
    print(f"Saved {output_path} (input 1x3x{imgsz}x{imgsz}, opset {opset})")


if __name__ == "__main__":
    tyro.cli(main)
