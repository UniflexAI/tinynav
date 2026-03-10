import tensorrt as trt
import numpy as np
import time
import cv2
from codetiming import Timer
import platform
import asyncio
from func import alru_cache_numpy

from cuda import cudart
import ctypes
import einops
import logging

numpy_to_ctypes = {
    np.dtype(np.float32): ctypes.c_float,
    np.dtype(np.int8):   ctypes.c_int8,
    np.dtype(np.uint8):  ctypes.c_uint8,
    np.dtype(np.int32):  ctypes.c_int32,
    np.dtype(np.int64):  ctypes.c_int64,
    np.dtype(np.bool_):  ctypes.c_bool
}

class TRTBase:
    def __init__(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        with Timer(name="[capture_graph]", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            self.graph_exec = self.capture_graph()
        logging.info(f"load {engine_path} done!")

    def _get_static_shape(self, name):
        """Return a concrete shape for a tensor, resolving dynamic dims via the profile if needed."""
        shape = tuple(self.context.get_tensor_shape(name))
        if -1 not in shape:
            return shape

        # Resolve from optimization profile (profile 0) when available.
        try:
            _, _, max_shape = self.engine.get_tensor_profile_shape(name, 0)
            return tuple(int(d) for d in max_shape)
        except Exception:
            # Fallback: replace dynamic dims with 1 to avoid crashes.
            return tuple(d if d != -1 else 1 for d in shape)

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        _, stream = cudart.cudaStreamCreate()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self._get_static_shape(name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            ctype_dtype = numpy_to_ctypes[dtype]
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            size = trt.volume(shape)
            nbytes = trt.volume(shape) * dtype.itemsize

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

            if is_input:
                inputs.append({"host": host_mem, "device": device_ptr, "shape": shape, "nbytes": nbytes})
            else:
                outputs.append({"host": host_mem, "device": device_ptr, "name": name, "nbytes": nbytes})

        return inputs, outputs, bindings, stream


    def capture_graph(self):
        # Ensure dynamic input shapes are specified before first execution.
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = self._get_static_shape(name)
                self.context.set_input_shape(name, shape)

        cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
        self.context.execute_async_v3(stream_handle=self.stream)

        _, graph = cudart.cudaStreamEndCapture(self.stream)
        _, graph_exec = cudart.cudaGraphInstantiate(graph, 0)
        cudart.cudaStreamSynchronize(self.stream)
        return graph_exec

    async def run_graph(self):
        if "aarch64" not in platform.machine():
            for inp in self.inputs:
                cudart.cudaMemcpyAsync(inp["device"], inp["host"].ctypes.data,
                                   inp["nbytes"],
                                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                   self.stream)

        cudart.cudaGraphLaunch(self.graph_exec, self.stream)

        if "aarch64" not in platform.machine():
            for out in self.outputs:
                cudart.cudaMemcpyAsync(out['host'].ctypes.data, out['device'], out['nbytes'], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)

        _, event = cudart.cudaEventCreate()
        cudart.cudaEventRecord(event, self.stream)
        while cudart.cudaEventQuery(event)[0] == cudart.cudaError_t.cudaErrorNotReady:
            await asyncio.sleep(0)

        results = {}
        for out in self.outputs:
            results[out["name"]] = out["host"].copy()
        return results


class SuperPointTRT(TRTBase):
    def __init__(self, engine_path=f"/tinynav/tinynav/models/superpoint_fp16_dynamic_{platform.machine()}.plan"):
        super().__init__(engine_path)
        # model input [1,1,H,W]
        self.input_shape = self.inputs[0]["shape"][2:4] # [H,W]

    # default threshold as
    # https://github.com/cvg/LightGlue/blob/746fac2c042e05d1865315b1413419f1c1e7ba55/lightglue/superpoint.py#L111
    #
    @alru_cache_numpy(maxsize=32)
    async def infer(self, input_image:np.ndarray, threshold = np.array([[0.0005]], dtype=np.float32)):
        # Resize to engine input size (may change aspect ratio for non-matching resolutions).
        h_in, w_in = input_image.shape[0], input_image.shape[1]
        h_net, w_net = self.input_shape[0], self.input_shape[1]
        image = cv2.resize(input_image, (w_net, h_net))
        image = image[None, None, :, :]

        np.copyto(self.inputs[0]["host"], image)
        np.copyto(self.inputs[1]["host"], threshold)

        results = await self.run_graph()

        # Scale keypoints from network coords (h_net, w_net) back to input image coords (h_in, w_in).
        # Use per-axis scale so Looper (640x544) and other resolutions match; img_shape is (width, height).
        scale_x = w_in / w_net
        scale_y = h_in / h_net
        k = results["kpts"][0]
        if k.shape[0] == 2:
            k[0] = (k[0] + 0.5) * scale_x - 0.5
            k[1] = (k[1] + 0.5) * scale_y - 0.5
        else:
            k[:, 0] = (k[:, 0] + 0.5) * scale_x - 0.5
            k[:, 1] = (k[:, 1] + 0.5) * scale_y - 0.5
        results["mask"] = results["mask"][:, :, None]
        return results

class LightGlueTRT(TRTBase):
    def __init__(self, engine_path=f"/tinynav/tinynav/models/lightglue_fp16_{platform.machine()}.plan"):
        super().__init__(engine_path)

    # default threshold as
    # https://github.com/cvg/LightGlue/blob/746fac2c042e05d1865315b1413419f1c1e7ba55/lightglue/lightglue.py#L333
    #
    @alru_cache_numpy(maxsize=32)
    async def infer(self, kpts0, kpts1, desc0, desc1, mask0, mask1, img_shape0, img_shape1, match_threshold = np.array([[0.1]], dtype=np.float32)):
        np.copyto(self.inputs[0]["host"], kpts0)
        np.copyto(self.inputs[1]["host"], kpts1)
        np.copyto(self.inputs[2]["host"], desc0)
        np.copyto(self.inputs[3]["host"], desc1)
        np.copyto(self.inputs[4]["host"], mask0)
        np.copyto(self.inputs[5]["host"], mask1)
        np.copyto(self.inputs[6]["host"], img_shape0)
        np.copyto(self.inputs[7]["host"], img_shape1)
        np.copyto(self.inputs[8]["host"], match_threshold)

        return await self.run_graph()

class Dinov2TRT(TRTBase):
    def __init__(self, engine_path=f"/tinynav/tinynav/models/dinov2_base_224x224_fp16_{platform.machine()}.plan"):
        super().__init__(engine_path)

    def preprocess_image(self, image, target_size=224):
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = einops.rearrange(image, "h w c-> 1 c h w")
        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = (image - mean) / std
        return image

    async def infer(self, image):
        image = self.preprocess_image(image)
        np.copyto(self.inputs[0]["host"], image)
        results = await self.run_graph()
        return results["last_hidden_state"][:, 0, :].squeeze(0)


# Retinify two-profile engine: (H, W) -> profile index
_STEREO_PROFILES = [(480, 848), (640, 544)]  # RealSense, Looper


def _alloc_pinned(shape, dtype):
    """Allocate pinned host buffer with exact shape (for profile-matched I/O)."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if "aarch64" in platform.machine():
        ptr = cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocMapped)[1]
    else:
        ptr = cudart.cudaMallocHost(nbytes)[1]
    ctype = numpy_to_ctypes[np.dtype(dtype)]
    arr = np.ctypeslib.as_array((ctype * int(np.prod(shape))).from_address(ptr))
    return arr.view(dtype).reshape(shape)


class StereoEngineTRT(TRTBase):
    def __init__(self, engine_path=f"/tinynav/tinynav/models/retinify_0_1_5_dynamic_{platform.machine()}.plan"):
        super().__init__(engine_path)
        # Profile 0 buffers: base class already allocated (1,1,480,848) for left/right and disp/depth
        h0, w0 = _STEREO_PROFILES[0]
        h1, w1 = _STEREO_PROFILES[1]
        sh0 = (1, 1, h0, w0)
        sh1 = (1, 1, h1, w1)
        self._inputs_p0 = self.inputs
        self._outputs_p0 = self.outputs
        left_1 = _alloc_pinned(sh1, np.uint8)
        right_1 = _alloc_pinned(sh1, np.uint8)
        disp_1 = _alloc_pinned(sh1, np.float32)
        depth_1 = _alloc_pinned(sh1, np.float32)
        self._inputs_p1 = [
            {"host": left_1, "device": self.inputs[0]["device"], "shape": sh1, "nbytes": left_1.nbytes},
            {"host": right_1, "device": self.inputs[1]["device"], "shape": sh1, "nbytes": right_1.nbytes},
            self.inputs[2],
            self.inputs[3],
        ]
        self._outputs_p1 = [
            {"host": disp_1, "device": self.outputs[0]["device"], "name": "disp", "nbytes": disp_1.nbytes},
            {"host": depth_1, "device": self.outputs[1]["device"], "name": "depth", "nbytes": depth_1.nbytes},
        ]
        self._current_profile = 0
        self._current_input_shapes = sh0

    def capture_graph(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])
        return None

    async def run_graph(self):
        profile = self._current_profile
        input_shapes = self._current_input_shapes
        inputs = self._inputs_p0 if profile == 0 else self._inputs_p1
        outputs = self._outputs_p0 if profile == 0 else self._outputs_p1
        if "aarch64" not in platform.machine():
            for inp in inputs:
                cudart.cudaMemcpyAsync(inp["device"], inp["host"].ctypes.data,
                                       inp["nbytes"],
                                       cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                       self.stream)
        self.context.set_optimization_profile_async(profile, self.stream)
        self.context.set_input_shape("left", input_shapes)
        self.context.set_input_shape("right", input_shapes)
        self.context.execute_async_v3(stream_handle=self.stream)
        if "aarch64" not in platform.machine():
            for out in outputs:
                cudart.cudaMemcpyAsync(out["host"].ctypes.data, out["device"], out["nbytes"],
                                       cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        _, event = cudart.cudaEventCreate()
        cudart.cudaEventRecord(event, self.stream)
        while cudart.cudaEventQuery(event)[0] == cudart.cudaError_t.cudaErrorNotReady:
            await asyncio.sleep(0)
        results = {}
        for out in outputs:
            results[out["name"]] = out["host"].copy()
        return results

    async def infer(self, left_img, right_img, baseline, focal_length):
        h_in, w_in = left_img.shape[0], left_img.shape[1]
        if (h_in, w_in) == (480, 848):
            profile, (h_net, w_net) = 0, (480, 848)
        elif (h_in, w_in) == (640, 544):
            profile, (h_net, w_net) = 1, (640, 544)
        else:
            h_net, w_net = _STEREO_PROFILES[0]
            profile = 0
            left_img = cv2.resize(left_img, (w_net, h_net))
            right_img = cv2.resize(right_img, (w_net, h_net))
            h_in, w_in = h_net, w_net

        self._current_profile = profile
        self._current_input_shapes = (1, 1, h_net, w_net)
        inputs = self._inputs_p0 if profile == 0 else self._inputs_p1
        left_tensor = left_img.astype(np.uint8)[None, None, :, :]
        right_tensor = right_img.astype(np.uint8)[None, None, :, :]
        np.copyto(inputs[0]["host"], left_tensor)
        np.copyto(inputs[1]["host"], right_tensor)
        np.copyto(inputs[2]["host"], baseline)
        np.copyto(inputs[3]["host"], focal_length)

        results = await self.run_graph()
        disp = results["disp"][0, 0, :, :]
        depth = results["depth"][0, 0, :, :]
        if (disp.shape[0], disp.shape[1]) != (h_in, w_in):
            disp = cv2.resize(disp, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        return disp.astype(np.float32), depth.astype(np.float32)

if __name__ == "__main__":
    dinov2 = Dinov2TRT()
    superpoint = SuperPointTRT()
    light_glue = LightGlueTRT()
    stereo_engine = StereoEngineTRT()

    # Test both RealSense and Looper resolutions (profile 0 and 1).
    # Each entry: (name, width, height)
    resolutions = [
        ("realsense", 848, 480),
        ("looper", 544, 640),
    ]

    match_threshold = np.array([0.1], dtype=np.float32)
    threshold = np.array([0.015], dtype=np.float32)

    for tag, width, height in resolutions:
        print(f"\n=== Testing stereo pipeline for {tag} resolution: {height}x{width} ===")
        image_shape = np.array([width, height], dtype=np.int64)

        dummy_left = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        dummy_right = np.random.randint(0, 256, (height, width), dtype=np.uint8)

        with Timer(text=f"[dinov2:{tag}] Elapsed time: {{milliseconds:.0f}} ms"):
            _ = asyncio.run(dinov2.infer(dummy_left))

        with Timer(text=f"[superpoint:{tag}] Elapsed time: {{milliseconds:.0f}} ms"):
            left_extract_result = asyncio.run(superpoint.infer(dummy_left))
            right_extract_result = asyncio.run(superpoint.infer(dummy_right))

        with Timer(text=f"[lightglue:{tag}] Elapsed time: {{milliseconds:.0f}} ms"):
            _ = asyncio.run(
                light_glue.infer(
                    left_extract_result["kpts"],
                    right_extract_result["kpts"],
                    left_extract_result["descps"],
                    right_extract_result["descps"],
                    left_extract_result["mask"],
                    right_extract_result["mask"],
                    image_shape,
                    image_shape,
                    match_threshold,
                )
            )

        with Timer(text=f"[stereo:{tag}] Elapsed time: {{milliseconds:.0f}} ms"):
            baseline = np.array([[0.05]], dtype=np.float32)
            focal_length = np.array([[323.0]], dtype=np.float32)
            _disp, _depth = asyncio.run(
                stereo_engine.infer(dummy_left, dummy_right, baseline, focal_length)
            )