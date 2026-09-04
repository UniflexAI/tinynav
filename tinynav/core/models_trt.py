import tensorrt as trt
import numpy as np
import cv2
from codetiming import Timer
import platform
import asyncio
import tyro
from tinynav.core.func import alru_cache_numpy

from cuda import cudart
import ctypes
import einops
import logging
from pathlib import Path

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
                inputs.append({"host": host_mem, "device": device_ptr, "shape": shape, "name": name, "nbytes": nbytes})
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


def save_matching_visualization(img0, img1, kpts0, kpts1, match_result, output_path="matching_vis.jpg"):
    keypoints0 = kpts0[0]
    keypoints1 = kpts1[0]
    match_indices = match_result["match_indices"][0]
    valid_mask = match_indices != -1
    matched0 = keypoints0[valid_mask]
    matched1 = keypoints1[match_indices[valid_mask]]

    match_vis = cv2.drawMatches(
        img0,
        [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in matched0],
        img1,
        [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in matched1],
        [cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(matched0))],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchesThickness=1,
    )
    cv2.imwrite(output_path, match_vis)
    print(f"Saved matching visualization to {output_path} (matches: {len(matched0)})")


def tag_output_path(output_path: str, tag: str) -> str:
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_{tag}{path.suffix}"))


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

    async def infer_global_and_patch_tokens(self, image):
        """Return the CLS descriptor and L2-normalized patch tokens in one run."""
        image = self.preprocess_image(image)
        np.copyto(self.inputs[0]["host"], image)
        results = await self.run_graph()
        hidden_state = results["last_hidden_state"].squeeze(0)
        global_embedding = hidden_state[0]
        tokens = hidden_state[1:]
        norms = np.linalg.norm(tokens, axis=1, keepdims=True)
        tokens = tokens / np.maximum(norms, 1e-8)
        return global_embedding, tokens.astype(np.float32)

    async def infer_patch_tokens(self, image):
        """Return L2-normalized DINOv2 patch tokens, excluding the CLS token."""
        image = self.preprocess_image(image)
        np.copyto(self.inputs[0]["host"], image)
        results = await self.run_graph()
        tokens = results["last_hidden_state"][:, 1:, :].squeeze(0)
        norms = np.linalg.norm(tokens, axis=1, keepdims=True)
        tokens = tokens / np.maximum(norms, 1e-8)
        return tokens.astype(np.float32)


# The 80 COCO class names yolo11n (and the rest of the YOLOv8/11 family) is
# trained on, index-matched to the class_id emitted by decode_yolo_output.
COCO_CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic_light",
    "fire_hydrant", "stop_sign", "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove", "skateboard", "surfboard",
    "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch",
    "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell_phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush",
)
_COCO_NAME_TO_ID = {name: i for i, name in enumerate(COCO_CLASS_NAMES)}


def coco_class_ids(names):
    """Resolve COCO class names (see COCO_CLASS_NAMES) to their class_id ints.

    Raises ValueError naming the offending entry on an unknown class name, so
    a typo in a config file fails loudly instead of silently keeping nothing.
    """
    try:
        return tuple(_COCO_NAME_TO_ID[name] for name in names)
    except KeyError as exc:
        raise ValueError(f"Unknown COCO class name: {exc.args[0]!r}") from exc


def letterbox_resize(image: np.ndarray, net_h: int, net_w: int, pad_value: int = 114):
    """Resize+pad `image` to (net_h, net_w) preserving aspect ratio.

    Returns (canvas, scale, (pad_left, pad_top)); orig = (net - pad) / scale
    maps a coordinate back from network space to the original image.
    """
    h, w = image.shape[:2]
    scale = min(net_w / w, net_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_left = (net_w - new_w) // 2
    pad_top = (net_h - new_h) // 2
    canvas = np.full((net_h, net_w, image.shape[2]), pad_value, dtype=image.dtype)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, (pad_left, pad_top)


def decode_yolo_output(raw, conf_threshold, iou_threshold, scale, pad):
    """Decode a raw (1, 4+num_classes, N) Ultralytics-style YOLO export tensor.

    `scale`/`pad` come from the matching `letterbox_resize` call and map
    network-space pixel coords back to the original image via
    orig = (net - pad) / scale. Kept free of TRT/ROS imports so it can be
    unit-tested with synthetic tensors.

    Returns a list of (class_id, score, x1, y1, x2, y2) in original-image
    pixel coordinates.
    """
    preds = np.asarray(raw, dtype=np.float32)
    if preds.ndim == 3:
        preds = preds[0]
    preds = preds.T  # (N, 4+num_classes)

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    keep = scores >= conf_threshold
    if not np.any(keep):
        return []
    boxes_xywh = boxes_xywh[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]

    pad_x, pad_y = pad
    cx = (boxes_xywh[:, 0] - pad_x) / scale
    cy = (boxes_xywh[:, 1] - pad_y) / scale
    w = boxes_xywh[:, 2] / scale
    h = boxes_xywh[:, 3] / scale
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0

    nms_boxes = np.stack([x1, y1, w, h], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), float(conf_threshold), float(iou_threshold))
    if len(indices) == 0:
        return []
    indices = np.asarray(indices).reshape(-1)

    detections = []
    for i in indices:
        bx1, by1, bw, bh = nms_boxes[i]
        detections.append((int(class_ids[i]), float(scores[i]), float(bx1), float(by1), float(bx1 + bw), float(by1 + bh)))
    return detections


class YoloDetectorTRT(TRTBase):
    """Closed-set COCO detector (Ultralytics YOLO export) used to flag
    "potential objects" for planning_node's object-voxel grid."""

    def __init__(
        self,
        engine_path=f"/tinynav/tinynav/models/yolo11n_640x640_{platform.machine()}.plan",
        confidence_threshold=0.4,
        iou_threshold=0.45,
    ):
        super().__init__(engine_path)
        if len(self.inputs) != 1:
            raise RuntimeError(f"YOLO engine must have 1 input, got {len(self.inputs)}")
        self.output_name = self.outputs[0]["name"]
        self.net_h = int(self.inputs[0]["shape"][2])
        self.net_w = int(self.inputs[0]["shape"][3])
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def preprocess(self, image: np.ndarray):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        canvas, scale, pad = letterbox_resize(image, self.net_h, self.net_w)
        tensor = einops.rearrange(canvas, "h w c -> 1 c h w").astype(np.float32) / 255.0
        return tensor, scale, pad

    async def infer(self, image: np.ndarray):
        tensor, scale, pad = self.preprocess(image)
        np.copyto(self.inputs[0]["host"], tensor.astype(self.inputs[0]["host"].dtype, copy=False))
        results = await self.run_graph()
        raw = np.asarray(results[self.output_name], dtype=np.float32)
        return decode_yolo_output(raw, self.confidence_threshold, self.iou_threshold, scale, pad)


class SigLIPImageTRT(TRTBase):
    def __init__(self, engine_path=f"/tinynav/tinynav/models/siglip_vit_b_16_webli_image_fp16_{platform.machine()}.plan"):
        super().__init__(engine_path)
        if len(self.inputs) != 1:
            raise RuntimeError(f"SigLIP image engine must have 1 input, got {len(self.inputs)}")
        self.input_shape = self.inputs[0]["shape"]
        self.output_name = self.outputs[0]["name"]
        self.net_h = int(self.input_shape[2])
        self.net_w = int(self.input_shape[3])

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] >= 3:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        image = cv2.resize(image, (self.net_w, self.net_h), interpolation=cv2.INTER_CUBIC)
        image = einops.rearrange(image, "h w c -> 1 c h w").astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
        return (image - mean) / std

    async def infer(self, image: np.ndarray) -> np.ndarray:
        image = self.preprocess_image(image)
        np.copyto(self.inputs[0]["host"], image.astype(self.inputs[0]["host"].dtype, copy=False))
        results = await self.run_graph()
        return np.asarray(results[self.output_name], dtype=np.float32).reshape(-1)


class SigLIPTextTRT(TRTBase):
    def __init__(
        self,
        engine_path=f"/tinynav/tinynav/models/siglip_vit_b_16_webli_text_fp16_{platform.machine()}.plan",
        tokenizer_path="/tinynav/tinynav/models/siglip_vit_b_16_webli_tokenizer.json",
    ):
        super().__init__(engine_path)
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError("SigLIP text retrieval requires the tokenizers package") from exc

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.output_name = self.outputs[0]["name"]
        self.input_by_name = {inp["name"]: inp for inp in self.inputs}
        self.context_length = int(self.inputs[0]["shape"][-1])

    def _tokenize(self, text: str) -> dict[str, np.ndarray]:
        encoded = self.tokenizer.encode(text)
        input_ids = encoded.ids[: self.context_length]
        attention_mask = [1] * len(input_ids)
        pad_len = self.context_length - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.token_to_id("[PAD]")
            if pad_id is None:
                pad_id = 0
            input_ids.extend([pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
        return {
            "input_ids": np.asarray(input_ids, dtype=np.int64)[None, :],
            "attention_mask": np.asarray(attention_mask, dtype=np.int64)[None, :],
        }

    async def infer(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        for inp in self.inputs:
            if inp["name"] in tokens:
                value = tokens[inp["name"]]
            elif len(self.inputs) == 1:
                value = tokens["input_ids"]
            else:
                raise RuntimeError(f"Unsupported SigLIP text engine input: {inp['name']}")
            np.copyto(inp["host"], value.astype(inp["host"].dtype, copy=False))
        results = await self.run_graph()
        return np.asarray(results[self.output_name], dtype=np.float32).reshape(-1)


class SigLIPTRT:
    def __init__(
        self,
        image_engine_path=f"/tinynav/tinynav/models/siglip_vit_b_16_webli_image_fp16_{platform.machine()}.plan",
        text_engine_path=f"/tinynav/tinynav/models/siglip_vit_b_16_webli_text_fp16_{platform.machine()}.plan",
        tokenizer_path="/tinynav/tinynav/models/siglip_vit_b_16_webli_tokenizer.json",
    ):
        self.image_engine_path = image_engine_path
        self.text_engine_path = text_engine_path
        self.tokenizer_path = tokenizer_path
        self.image_encoder = None
        self.text_encoder = None

    async def encode_image(self, image: np.ndarray) -> np.ndarray:
        if self.image_encoder is None:
            self.image_encoder = SigLIPImageTRT(self.image_engine_path)
        return await self.image_encoder.infer(image)

    async def encode_text(self, text: str) -> np.ndarray:
        if self.text_encoder is None:
            self.text_encoder = SigLIPTextTRT(self.text_engine_path, self.tokenizer_path)
        return await self.text_encoder.infer(text)


class FoundationStereoTRT(TRTBase):
    def __init__(
        self,
        engine_path=f"/tinynav/tinynav/models/foundation_stereo_11-33-40_256x320_4_{platform.machine()}.plan",
    ):
        super().__init__(engine_path)
        if len(self.inputs) != 2:
            raise RuntimeError(f"FoundationStereo engine must have 2 inputs, got {len(self.inputs)}")
        if len(self.outputs) != 1:
            raise RuntimeError(f"FoundationStereo engine must have 1 output, got {len(self.outputs)}")

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
        image = image.astype(np.float32, copy=False)
        return np.transpose(image, (2, 0, 1))[None, ...]

    def _resize_for_engine(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == (self.net_h, self.net_w):
            return image
        return cv2.resize(image, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR)

    async def infer(self, left_img, right_img, baseline, focal_length):
        if left_img.shape[:2] != right_img.shape[:2]:
            raise ValueError(f"Left/right shape mismatch: {left_img.shape} vs {right_img.shape}")

        h_in, w_in = left_img.shape[:2]
        left_tensor = self._to_three_channel_float(self._resize_for_engine(left_img))
        right_tensor = self._to_three_channel_float(self._resize_for_engine(right_img))

        np.copyto(self.inputs[self.left_idx]["host"], left_tensor)
        np.copyto(self.inputs[self.right_idx]["host"], right_tensor)

        results = await self.run_graph()
        disp_net = np.asarray(results[self.output_name], dtype=np.float32).reshape(self.net_h, self.net_w)
        disp_net = np.clip(disp_net, 0.0, None)
        if (h_in, w_in) == (self.net_h, self.net_w):
            disp = disp_net
        else:
            disp = cv2.resize(disp_net, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
            disp *= float(w_in) / float(self.net_w)
            disp = np.clip(disp, 0.0, None)

        # hack
        disp[300:,:] = 0.0

        depth = disparity_to_depth(disp, baseline, focal_length)
        return disp, depth


class RetinifyTRT(TRTBase):
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
                f"RetinifyTRT output shape mismatch: got disp {disp.shape}, expected ({h_in}, {w_in})"
            )
        disp = disp.astype(np.float32)
        depth = disparity_to_depth(disp, baseline, focal_length)
        return disp, depth


StereoEngineTRT = RetinifyTRT


def main(
    left: str | None = None,
    right: str | None = None,
    output: str = "matching_vis.jpg",
) -> None:
    if (left is None) != (right is None):
        raise ValueError("--left and --right must be provided together")

    dinov2 = Dinov2TRT()
    superpoint = SuperPointTRT()
    light_glue = LightGlueTRT()
    stereo_engine = StereoEngineTRT()

    image_pairs = []
    if left is not None:
        left_img = cv2.imread(left, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right, cv2.IMREAD_GRAYSCALE)
        if left_img is None:
            raise FileNotFoundError(f"failed to read left image: {left}")
        if right_img is None:
            raise FileNotFoundError(f"failed to read right image: {right}")
        if left_img.shape != right_img.shape:
            raise ValueError(f"left/right image shapes differ: {left_img.shape} vs {right_img.shape}")
        image_pairs.append(("input", left_img, right_img))
    else:
        # Synthetic sanity test for both RealSense and Looper resolutions.
        for tag, width, height in [
            ("realsense", 848, 480),
            ("looper", 544, 640),
        ]:
            left_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            right_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            image_pairs.append((tag, left_img, right_img))

    match_threshold = np.array([0.1], dtype=np.float32)
    save_tagged_outputs = len(image_pairs) > 1
    for tag, dummy_left, dummy_right in image_pairs:
        height, width = dummy_left.shape[:2]
        print(f"\n=== Testing stereo pipeline for {tag} resolution: {height}x{width} ===")
        image_shape = np.array([width, height], dtype=np.int64)

        with Timer(text=f"[dinov2:{tag}] Elapsed time: {{milliseconds:.0f}} ms"):
            _ = asyncio.run(dinov2.infer(dummy_left))

        with Timer(text=f"[superpoint:{tag}] Elapsed time: {{milliseconds:.0f}} ms"):
            left_extract_result = asyncio.run(superpoint.infer(dummy_left))
            right_extract_result = asyncio.run(superpoint.infer(dummy_right))

        with Timer(text=f"[lightglue:{tag}] Elapsed time: {{milliseconds:.0f}} ms"):
            matching_result = asyncio.run(
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
            save_matching_visualization(
                dummy_left,
                dummy_right,
                left_extract_result["kpts"],
                right_extract_result["kpts"],
                matching_result,
                tag_output_path(output, tag) if save_tagged_outputs else output,
            )

        with Timer(text=f"[stereo:{tag}] Elapsed time: {{milliseconds:.0f}} ms"):
            baseline = np.array([[0.05]], dtype=np.float32)
            focal_length = np.array([[323.0]], dtype=np.float32)
            _disp, _depth = asyncio.run(
                stereo_engine.infer(dummy_left, dummy_right, baseline, focal_length)
            )


if __name__ == "__main__":
    tyro.cli(main)
