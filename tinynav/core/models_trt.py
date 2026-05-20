try:
    import tensorrt as trt
except ImportError:
    trt = None
import numpy as np
import cv2
from codetiming import Timer
import platform
import asyncio
from tinynav.core.func import alru_cache_numpy

try:
    from cuda import cudart
except ImportError:
    cudart = None
import ctypes
import einops
import logging
from typing import Any

logger = logging.getLogger(__name__)

numpy_to_ctypes = {
    np.dtype(np.float32): ctypes.c_float,
    np.dtype(np.float16): ctypes.c_uint16,
    np.dtype(np.int8):   ctypes.c_int8,
    np.dtype(np.uint8):  ctypes.c_uint8,
    np.dtype(np.int32):  ctypes.c_int32,
    np.dtype(np.int64):  ctypes.c_int64,
    np.dtype(np.bool_):  ctypes.c_bool
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
    def __init__(self, engine_path):
        if trt is None or cudart is None:
            raise ImportError(
                "tensorrt and cuda-python are required for TRT models. "
                "Use BoW mode or install TensorRT Python bindings."
            )
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


class ORBFeatureTRTCompatible:
    """
    ORB feature extractor with a SuperPoint-compatible output interface.

    Returns a dict with:
      - kpts:  (1, N, 2) float32
      - descps: (1, N, 32) float32
      - mask:  (1, N, 1) float32
    """

    def __init__(self, nfeatures: int = 1024):
        self.nfeatures = int(nfeatures)
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures)

    async def infer(self, input_image: np.ndarray):
        if input_image is None:
            raise ValueError("input_image is None")
        if input_image.ndim == 3:
            gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        elif input_image.ndim == 2:
            gray = input_image
        else:
            raise ValueError(f"Unsupported input_image ndim={input_image.ndim}")

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if keypoints is None or len(keypoints) == 0:
            return {
                "kpts": np.zeros((1, 0, 2), dtype=np.float32),
                "descps": np.zeros((1, 0, 32), dtype=np.float32),
                "mask": np.zeros((1, 0, 1), dtype=np.float32),
            }

        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        if descriptors is None:
            descps = np.zeros((kpts.shape[0], 32), dtype=np.float32)
        else:
            descps = descriptors.astype(np.float32) / 255.0

        return {
            "kpts": kpts[None, :, :],
            "descps": descps[None, :, :],
            "mask": np.ones((1, kpts.shape[0], 1), dtype=np.float32),
        }


class ORBMatcher:
    """
    ORB descriptor matcher with BFMatcher(crossCheck=True) and geometric RANSAC filtering.

    Output is LightGlue-compatible:
      - match_indices: (1, N0) int32, each value is matched index in keypoints1 or -1.
    """

    def __init__(self, ransac_reproj_threshold: float = 1.0):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ransac_reproj_threshold = float(ransac_reproj_threshold)

    async def infer(
        self,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        desc0: np.ndarray,
        desc1: np.ndarray,
        mask0: np.ndarray,
        mask1: np.ndarray,
        img_shape0=None,
        img_shape1=None,
        match_threshold=None,
    ):
        k0 = np.asarray(kpts0[0], dtype=np.float32)
        k1 = np.asarray(kpts1[0], dtype=np.float32)
        if k0.ndim != 2 or k1.ndim != 2 or k0.shape[1] < 2 or k1.shape[1] < 2:
            n0 = int(k0.shape[0]) if k0.ndim >= 1 else 0
            return {"match_indices": np.full((1, n0), -1, dtype=np.int32)}
        # keep only xy coordinates
        k0 = k0[:, :2]
        k1 = k1[:, :2]
        n0 = int(k0.shape[0])
        out = {"match_indices": np.full((1, n0), -1, dtype=np.int32)}
        if n0 == 0 or k1.shape[0] == 0:
            return out

        d0 = np.asarray(desc0[0], dtype=np.float32)
        d1 = np.asarray(desc1[0], dtype=np.float32)
        if d0.ndim != 2 or d1.ndim != 2 or d0.shape[0] == 0 or d1.shape[0] == 0:
            return out
        # Ensure descriptor rows and keypoints rows are aligned.
        n0_valid = min(k0.shape[0], d0.shape[0])
        n1_valid = min(k1.shape[0], d1.shape[0])
        if n0_valid == 0 or n1_valid == 0:
            return out
        k0 = k0[:n0_valid]
        k1 = k1[:n1_valid]
        d0 = d0[:n0_valid]
        d1 = d1[:n1_valid]
        out = {"match_indices": np.full((1, n0_valid), -1, dtype=np.int32)}

        m0 = np.asarray(mask0[0, :, 0] > 0, dtype=bool) if mask0.size else np.ones((d0.shape[0],), dtype=bool)
        m1 = np.asarray(mask1[0, :, 0] > 0, dtype=bool) if mask1.size else np.ones((d1.shape[0],), dtype=bool)
        if m0.shape[0] != d0.shape[0]:
            m0 = np.ones((d0.shape[0],), dtype=bool)
        if m1.shape[0] != d1.shape[0]:
            m1 = np.ones((d1.shape[0],), dtype=bool)

        idx0 = np.where(m0)[0]
        idx1 = np.where(m1)[0]
        if idx0.size == 0 or idx1.size == 0:
            return out

        # ORB descriptor is binary; convert normalized float [0,1] back to uint8 [0,255].
        d0_u8 = np.clip(np.rint(d0[idx0] * 255.0), 0, 255).astype(np.uint8)
        d1_u8 = np.clip(np.rint(d1[idx1] * 255.0), 0, 255).astype(np.uint8)
        if d0_u8.size == 0 or d1_u8.size == 0:
            return out

        raw_matches = self.bf.match(d0_u8, d1_u8)
        if len(raw_matches) == 0:
            return out

        raw_matches = sorted(raw_matches, key=lambda m: m.distance)
        pts0 = np.array([k0[idx0[m.queryIdx]] for m in raw_matches], dtype=np.float32).reshape(-1, 2)
        pts1 = np.array([k1[idx1[m.trainIdx]] for m in raw_matches], dtype=np.float32).reshape(-1, 2)

        inlier_mask = np.ones((len(raw_matches),), dtype=bool)
        if len(raw_matches) >= 8 and np.isfinite(pts0).all() and np.isfinite(pts1).all():
            logger.info(
                "ORBMatcher RANSAC input: pts0.shape=%s pts1.shape=%s raw_matches=%d",
                pts0.shape,
                pts1.shape,
                len(raw_matches),
            )
            _, ransac_inliers = cv2.findFundamentalMat(
                pts0,
                pts1,
                cv2.FM_RANSAC,
                self.ransac_reproj_threshold,
                0.99,
            )
            if ransac_inliers is not None and ransac_inliers.size == len(raw_matches):
                inlier_mask = ransac_inliers.reshape(-1).astype(bool)

        for i, m in enumerate(raw_matches):
            if not inlier_mask[i]:
                continue
            gi = int(idx0[m.queryIdx])
            gj = int(idx1[m.trainIdx])
            out["match_indices"][0, gi] = gj
        return out


class DBoW3Engine:
    """
    Thin Python wrapper for PyDBoW3 database operations.

    Supports adding/querying either:
      - ORBFeatureTRTCompatible outputs (dict with "descps")
      - Raw descriptor arrays shaped (N, 32), dtype uint8/float32
    """

    def __init__(self, vocabulary_path: str):
        self._bow = self._import_module()
        self.voc = self._bow.Vocabulary()
        load_ret = self.voc.load(vocabulary_path)
        # PyDBoW3 often returns None on success (instead of True).
        if load_ret is False:
            raise RuntimeError(f"Failed to load DBoW3 vocabulary: {vocabulary_path}")
        self.db = self._bow.Database()
        self.db.setVocabulary(self.voc)

    @staticmethod
    def _import_module():
        try:
            import pydbow3 as bow  # type: ignore
            return bow
        except Exception:
            import pyDBoW3 as bow  # type: ignore
            return bow

    @staticmethod
    def _normalize_desc(features: Any) -> np.ndarray:
        if isinstance(features, dict):
            if "descps" not in features:
                raise ValueError("features dict must contain 'descps'")
            desc = np.asarray(features["descps"])
            if desc.ndim == 3:
                desc = desc[0]
        else:
            desc = np.asarray(features)

        if desc.ndim != 2:
            raise ValueError(f"Descriptor array must be 2D, got shape {desc.shape}")
        if desc.shape[1] != 32:
            raise ValueError(f"Expected ORB descriptor width 32, got {desc.shape[1]}")
        if desc.size == 0:
            return np.zeros((0, 32), dtype=np.uint8)

        if desc.dtype == np.uint8:
            return np.ascontiguousarray(desc)
        # ORBFeatureTRTCompatible emits float32 in [0,1], convert back to binary-bytes domain.
        if np.issubdtype(desc.dtype, np.floating):
            return np.ascontiguousarray(np.clip(np.rint(desc * 255.0), 0, 255).astype(np.uint8))
        return np.ascontiguousarray(desc.astype(np.uint8))

    def add(self, features: Any):
        desc = self._normalize_desc(features)
        return self.db.add(desc)

    def query(self, features: Any, max_results: int = 10):
        desc = self._normalize_desc(features)
        try:
            results = self.db.query(desc, int(max_results))
        except TypeError:
            # Some bindings expose query(desc) only.
            results = self.db.query(desc)
        return [
            {
                "id": int(getattr(r, "Id", getattr(r, "id", -1))),
                "score": float(getattr(r, "Score", getattr(r, "score", 0.0))),
            }
            for r in results
        ]


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


class StereoEngineTRT(TRTBase):
    def _get_static_shape(self, name):
        """Ensure the stereo output gets a valid max shape for buffer allocation.

        Retinify is disp-only with NHWC tensors (B, H, W, C). Some TensorRT
        versions report dynamic outputs with empty/scalar shapes. Instead of
        asking output profile shape directly, derive max output shape from the
        "left" input profile because output shares the same spatial resolution.
        """
        if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            try:
                _, _, max_in_shape = self.engine.get_tensor_profile_shape("left", 0)
                # left input is NHWC -> output is NHWC with single channel.
                return (1, int(max_in_shape[1]), int(max_in_shape[2]), 1)
            except Exception:
                pass
        return super()._get_static_shape(name)

    def __init__(self, engine_path=f"/tinynav/tinynav/models/retinify_0_1_5_dynamic_{platform.machine()}.plan"):
        super().__init__(engine_path)
        if len(self.inputs) != 2:
            raise RuntimeError(f"Retinify disp-only engine must have 2 inputs, got {len(self.inputs)}")
        if len(self.outputs) != 1:
            raise RuntimeError(f"Retinify disp-only engine must have 1 output, got {len(self.outputs)}")
        self.output_name = self.outputs[0]["name"]
        self.input_dtype = self.inputs[0]["host"].dtype
        # Current shapes/byte sizes are set per infer() call, based on the
        # actually received image size (H, W), not the engine's max profile.
        self._current_input_shapes = (1, 1, 1, 1)
        self._current_input_nbytes = 0

    def capture_graph(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])
        return None

    async def run_graph(self):
        input_shapes = self._current_input_shapes
        if "aarch64" not in platform.machine():
            cudart.cudaMemcpyAsync(self.inputs[0]["device"], self.inputs[0]["host"].ctypes.data,
                                   self._current_input_nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            cudart.cudaMemcpyAsync(self.inputs[1]["device"], self.inputs[1]["host"].ctypes.data,
                                   self._current_input_nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self.context.set_optimization_profile_async(0, self.stream)
        self.context.set_input_shape("left", input_shapes)
        self.context.set_input_shape("right", input_shapes)
        self.context.execute_async_v3(stream_handle=self.stream)
        h_net, w_net = input_shapes[1], input_shapes[2]
        if "aarch64" not in platform.machine():
            for out in self.outputs:
                nbytes = h_net * w_net * np.float32().itemsize
                cudart.cudaMemcpyAsync(
                    out["host"].ctypes.data,
                    out["device"],
                    nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                )
        cudart.cudaStreamSynchronize(self.stream)
        results = {}
        for out in self.outputs:
            flat = np.asarray(out["host"]).reshape(-1)
            needed = h_net * w_net
            results[out["name"]] = flat[:needed].reshape(h_net, w_net).copy()
        return results

    async def infer(self, left_img, right_img, baseline, focal_length):
        h_in, w_in = left_img.shape[0], left_img.shape[1]

        self._current_input_shapes = (1, h_in, w_in, 1)
        # Retinify ONNX takes FLOAT inputs in NHWC layout.
        left_tensor = left_img.astype(self.input_dtype, copy=False)[None, :, :, None]
        right_tensor = right_img.astype(self.input_dtype, copy=False)[None, :, :, None]
        self._current_input_nbytes = left_tensor.nbytes

        # Copy only the active region into max-profile host buffers.
        np.copyto(self.inputs[0]["host"].reshape(-1)[: left_tensor.size], left_tensor.reshape(-1))
        np.copyto(self.inputs[1]["host"].reshape(-1)[: right_tensor.size], right_tensor.reshape(-1))

        results = await self.run_graph()
        disp = results[self.output_name]
        if disp.shape != (h_in, w_in):
            raise RuntimeError(
                f"StereoEngine output shape mismatch: got disp {disp.shape}, expected ({h_in}, {w_in})"
            )
        disp = disp.astype(np.float32)
        depth = disparity_to_depth(disp, baseline, focal_length)
        return disp, depth


if __name__ == "__main__":
    # Synthetic sanity test for both RealSense and Looper resolutions.
    dinov2 = Dinov2TRT()
    superpoint = SuperPointTRT()
    light_glue = LightGlueTRT()
    stereo_engine = StereoEngineTRT()

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
