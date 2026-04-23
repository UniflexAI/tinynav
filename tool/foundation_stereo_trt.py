import argparse
import asyncio
import ctypes
import logging
import platform
import sys

import cv2
import numpy as np
import tensorrt as trt
from codetiming import Timer
from cuda import cudart

try:
    import rclpy
    from cv_bridge import CvBridge
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object

numpy_to_ctypes = {
    np.dtype(np.float32): ctypes.c_float,
    np.dtype(np.float16): ctypes.c_uint16,
    np.dtype(np.int8): ctypes.c_int8,
    np.dtype(np.uint8): ctypes.c_uint8,
    np.dtype(np.int32): ctypes.c_int32,
    np.dtype(np.int64): ctypes.c_int64,
    np.dtype(np.bool_): ctypes.c_bool,
}


def disparity_to_depth(disparity: np.ndarray, baseline: float, focal_length: float) -> np.ndarray:
    disparity = np.asarray(disparity, dtype=np.float32)
    baseline = float(np.asarray(baseline).reshape(-1)[0])
    focal_length = float(np.asarray(focal_length).reshape(-1)[0])

    if baseline <= 0.0:
        raise ValueError(f"baseline must be positive, got {baseline}")
    if focal_length <= 0.0:
        raise ValueError(f"focal_length must be positive, got {focal_length}")

    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0.0)
    depth[valid] = (baseline * focal_length) / disparity[valid]
    return depth


class TRTBase:
    def __init__(self, engine_path: str):
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        _, stream = cudart.cudaStreamCreate()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            ctype_dtype = numpy_to_ctypes[dtype]
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            size = trt.volume(shape)
            nbytes = size * dtype.itemsize

            if "aarch64" in platform.machine():
                ptr = cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocMapped)[1]
                host_mem = np.ctypeslib.as_array((ctype_dtype * size).from_address(ptr))
                host_mem = host_mem.view(dtype).reshape(shape)
                device_ptr = cudart.cudaHostGetDevicePointer(ptr, 0)[1]
            else:
                ptr = cudart.cudaMallocHost(nbytes)[1]
                host_mem = np.ctypeslib.as_array((ctype_dtype * size).from_address(ptr))
                host_mem = host_mem.view(dtype).reshape(shape)
                device_ptr = cudart.cudaMalloc(nbytes)[1]

            bindings.append(int(device_ptr))
            io = {
                "name": name,
                "host": host_mem,
                "device": device_ptr,
                "shape": shape,
                "nbytes": nbytes,
            }
            if is_input:
                inputs.append(io)
            else:
                outputs.append(io)

        return inputs, outputs, bindings, stream

    async def run_graph(self):
        if "aarch64" not in platform.machine():
            for inp in self.inputs:
                cudart.cudaMemcpyAsync(
                    inp["device"],
                    inp["host"].ctypes.data,
                    inp["nbytes"],
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
        self.context.execute_async_v3(stream_handle=self.stream)

        if "aarch64" not in platform.machine():
            for out in self.outputs:
                cudart.cudaMemcpyAsync(
                    out["host"].ctypes.data,
                    out["device"],
                    out["nbytes"],
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                )

        _, event = cudart.cudaEventCreate()
        cudart.cudaEventRecord(event, self.stream)
        while cudart.cudaEventQuery(event)[0] == cudart.cudaError_t.cudaErrorNotReady:
            await asyncio.sleep(0)

        return {out["name"]: out["host"].copy() for out in self.outputs}


class FoundationStereoTRT(TRTBase):
    def __init__(self, engine_path=f"/tinynav/tinynav/models/foundation_stereo_{platform.machine()}.plan"):
        super().__init__(engine_path)
        if len(self.inputs) != 2:
            raise RuntimeError(f"Expected 2 inputs (left/right), got {len(self.inputs)}")
        if len(self.outputs) != 1:
            raise RuntimeError(f"Expected 1 output (disp), got {len(self.outputs)}")

        self.left_idx = 0 if self.inputs[0]["name"] == "left" else 1
        self.right_idx = 1 - self.left_idx
        self.output_name = self.outputs[0]["name"]

        self.net_h = int(self.inputs[self.left_idx]["shape"][2])
        self.net_w = int(self.inputs[self.left_idx]["shape"][3])

    def _to_three_channel_float(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim == 3 and image.shape[2] >= 3:
            image = image[:, :, :3]
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        image = cv2.resize(image, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32, copy=False)
        return np.transpose(image, (2, 0, 1))[None, ...]

    async def infer(self, left_img, right_img, baseline, focal_length):
        if left_img.shape[:2] != right_img.shape[:2]:
            raise ValueError(f"Left/right shape mismatch: {left_img.shape} vs {right_img.shape}")

        h_in, w_in = left_img.shape[:2]
        scale_x = self.net_w / float(w_in)

        left_tensor = self._to_three_channel_float(left_img)
        right_tensor = self._to_three_channel_float(right_img)

        np.copyto(self.inputs[self.left_idx]["host"], left_tensor)
        np.copyto(self.inputs[self.right_idx]["host"], right_tensor)

        results = await self.run_graph()

        disp_net = np.asarray(results[self.output_name], dtype=np.float32).reshape(self.net_h, self.net_w)
        disp_net = np.clip(disp_net, 0.0, None)

        if (h_in, w_in) != (self.net_h, self.net_w):
            disp = cv2.resize(disp_net, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
            disp = disp / scale_x
        else:
            disp = disp_net

        depth = disparity_to_depth(disp, baseline, focal_length)
        return disp, depth


class FoundationStereoDepthNode(Node):
    def __init__(
        self,
        engine_path=f"/tinynav/tinynav/models/foundation_stereo_{platform.machine()}.plan",
        left_topic="/camera/camera/infra1/image_rect_raw",
        right_topic="/camera/camera/infra2/image_rect_raw",
        camera_info_topic="/camera/camera/infra2/camera_info",
        keyframe_depth_topic="/keyframe/depth",
        queue_size=10,
        slop=0.02,
    ):
        super().__init__("foundation_stereo_depth_node")
        self.logger = logging.getLogger(__name__)
        self.bridge = CvBridge()
        self.stereo_engine = FoundationStereoTRT(engine_path=engine_path)

        self.baseline = None
        self.focal_length = None
        self.camera_info_msg = None

        self.camerainfo_sub = self.create_subscription(CameraInfo, camera_info_topic, self.info_callback, 10)
        self.left_sub = Subscriber(self, Image, left_topic)
        self.right_sub = Subscriber(self, Image, right_topic)
        self.ts = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=queue_size, slop=slop)
        self.ts.registerCallback(self.images_callback)

        self.keyframe_depth_pub = self.create_publisher(Image, keyframe_depth_topic, 10)
        self.logger.info(
            "FoundationStereoDepthNode initialized. left=%s right=%s info=%s depth=%s",
            left_topic,
            right_topic,
            camera_info_topic,
            keyframe_depth_topic,
        )

    def info_callback(self, msg: CameraInfo):
        if self.focal_length is not None:
            return
        k = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        fx = float(k[0, 0])
        tx = float(msg.p[3])
        baseline = -tx / fx
        if baseline <= 0.0:
            self.get_logger().warning(f"Baseline from CameraInfo is non-positive: {baseline:.6f}")
        self.focal_length = fx
        self.baseline = baseline
        self.camera_info_msg = msg
        self.get_logger().info(
            f"Camera intrinsics received. fx={self.focal_length:.3f} baseline={self.baseline:.5f} m"
        )
        self.destroy_subscription(self.camerainfo_sub)

    def images_callback(self, left_msg: Image, right_msg: Image):
        if self.focal_length is None or self.baseline is None:
            self.get_logger().warning("Skipping stereo frame: waiting for camera_info.")
            return

        left_img = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
        right_img = self.bridge.imgmsg_to_cv2(right_msg, "mono8")

        with Timer(
            name="FoundationStereo Inference",
            text="[{name}] Elapsed time: {milliseconds:.0f} ms",
            logger=self.logger.info,
        ):
            _, depth = asyncio.run(
                self.stereo_engine.infer(
                    left_img,
                    right_img,
                    np.array([[self.baseline]], dtype=np.float32),
                    np.array([[self.focal_length]], dtype=np.float32),
                )
            )

        with Timer(
            name="Publish Keyframe Depth",
            text="[{name}] Elapsed time: {milliseconds:.0f} ms",
            logger=self.logger.info,
        ):
            depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
            depth_msg.header.stamp = left_msg.header.stamp
            depth_msg.header.frame_id = "camera"
            self.keyframe_depth_pub.publish(depth_msg)


def main(args=None):
    if not ROS_AVAILABLE:
        raise RuntimeError("ROS dependencies are unavailable. Please run in the TinyNav ROS2 environment.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine_path",
        type=str,
        default=f"/tinynav/tinynav/models/foundation_stereo_{platform.machine()}.plan",
    )
    parser.add_argument("--left_topic", type=str, default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--right_topic", type=str, default="/camera/camera/infra2/image_rect_raw")
    parser.add_argument("--camera_info_topic", type=str, default="/camera/camera/infra2/camera_info")
    parser.add_argument("--keyframe_depth_topic", type=str, default="/keyframe/depth")
    parser.add_argument("--sync_queue_size", type=int, default=10)
    parser.add_argument("--sync_slop", type=float, default=0.02)
    parser.add_argument("--log_level", type=str, default="INFO")
    parsed_args, _ = parser.parse_known_args(args if args is not None else sys.argv[1:])

    logging.basicConfig(
        level=getattr(logging, parsed_args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    rclpy.init(args=args)
    node = FoundationStereoDepthNode(
        engine_path=parsed_args.engine_path,
        left_topic=parsed_args.left_topic,
        right_topic=parsed_args.right_topic,
        camera_info_topic=parsed_args.camera_info_topic,
        keyframe_depth_topic=parsed_args.keyframe_depth_topic,
        queue_size=parsed_args.sync_queue_size,
        slop=parsed_args.sync_slop,
    )
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    import os

    left_path = "/tinynav/tests/data/looper/left.png"
    right_path = "/tinynav/tests/data/looper/right.png"
    run_local_infer = (
        len(sys.argv) > 1 and sys.argv[1] == "--local_infer"
    ) or (not ROS_AVAILABLE and os.path.exists(left_path) and os.path.exists(right_path))
    if run_local_infer:
        left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        model = FoundationStereoTRT()
        disp, depth = asyncio.run(
            model.infer(
                left,
                right,
                np.array([[0.056]], dtype=np.float32),
                np.array([[323.0]], dtype=np.float32),
            )
        )
        print("disp", disp.shape, disp.dtype, float(np.nanmean(disp)))
        print("depth", depth.shape, depth.dtype, float(np.nanmean(depth[np.isfinite(depth)])))
    else:
        main()
