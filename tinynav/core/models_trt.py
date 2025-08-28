import tensorrt as trt
import numpy as np
import cv2
from codetiming import Timer
import platform
import asyncio
from async_lru import alru_cache

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

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        _, stream = cudart.cudaStreamCreate()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(name)
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
    def __init__(self, engine_path=f"/tinynav/tinynav/models/superpoint_240x424_fp16_{platform.machine()}.plan"):
        super().__init__(engine_path)
        # model input [1,1,H,W]
        self.input_shape = self.inputs[0]["shape"][2:4] # [H,W]

    # default threshold as 
    # https://github.com/cvg/LightGlue/blob/746fac2c042e05d1865315b1413419f1c1e7ba55/lightglue/superpoint.py#L111
    #
    async def infer(self, input_image:np.ndarray, threshold = np.array([[0.0005]], dtype=np.float32)):
        # resize to input_size
        scale = self.input_shape[0] / input_image.shape[0]
        image = cv2.resize(input_image, (self.input_shape[1], self.input_shape[0]))
        image = image[None, None, :, :]

        np.copyto(self.inputs[0]["host"], image)
        np.copyto(self.inputs[1]["host"], threshold)

        results = await self.run_graph()

        results["kpts"][0] = (results["kpts"][0] + 0.5) / scale - 0.5
        results["mask"] = results["mask"][:, :, None]
        return results

    async def memorized_infer(self, input_image:np.ndarray, threshold = np.array([0.0005], dtype=np.float32)):
        input_bytes = input_image.tobytes()
        return await self.infer_cached(input_bytes, input_image.shape, input_image.dtype.str, threshold.item())

    @alru_cache(maxsize=128)
    async def infer_cached(self, input_bytes, shape, dtype_str, threshold):
        input_image = np.frombuffer(input_bytes, dtype=dtype_str).reshape(shape)
        return await self.infer(input_image, np.array([threshold]))


class LightGlueTRT(TRTBase):
    def __init__(self, engine_path=f"/tinynav/tinynav/models/lightglue_fp16_{platform.machine()}.plan"):
        super().__init__(engine_path)

    # default threshold as
    # https://github.com/cvg/LightGlue/blob/746fac2c042e05d1865315b1413419f1c1e7ba55/lightglue/lightglue.py#L333
    #
    async def infer(self, kpts0, kpts1, desc0, desc1, mask0, mask1, img_shape0, img_shape1, match_threshold = np.array([[0.1]])):
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
    def __init__(self, engine_path=f"/tinynav/tinynav/models/retinify_0_1_4_480x848_{platform.machine()}.plan"):
        super().__init__(engine_path)

    async def infer(self, left_img, right_img):
        left_tensor = left_img.astype(np.float32)[None,:,:,None]
        right_tensor = right_img.astype(np.float32)[None,:,:,None]

        np.copyto(self.inputs[0]["host"], left_tensor)
        np.copyto(self.inputs[1]["host"], right_tensor)

        results = await self.run_graph()

        # left right consistency check
        out_map = results['disparity'][0, :, :, 0]
        yy, xx = np.meshgrid(np.arange(out_map.shape[0]), np.arange(out_map.shape[1]), indexing='ij')
        invalid = (xx - out_map) < 0
        out_map[invalid] = np.inf
        return out_map

if __name__ == "__main__":
    dinov2 = Dinov2TRT()
    superpoint = SuperPointTRT()
    light_glue = LightGlueTRT()
    stereo_engine = StereoEngineTRT()

    # Create dummy zero inputs
    image_shape = np.array([848, 480], dtype=np.int64)
    width, height = image_shape
    match_threshold = np.array([0.1], dtype=np.float32)
    threshold = np.array([0.015], dtype=np.float32)

    dummy_left = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    dummy_right = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    image = dinov2.preprocess_image(dummy_left)
    with Timer(text="[dinov2] Elapsed time: {milliseconds:.0f} ms"):
        embedding = asyncio.run(dinov2.infer(image))["last_hidden_state"][:, 0, :]
        embedding = np.squeeze(embedding, axis=0)

    with Timer(text="[superpoint] Elapsed time: {milliseconds:.0f} ms"):
        left_extract_result = asyncio.run(superpoint.infer(dummy_left))
        right_extract_result = asyncio.run(superpoint.infer(dummy_right))

    with Timer(text="[lightglue] Elapsed time: {milliseconds:.0f} ms"):
        match_result = asyncio.run(light_glue.infer(
            left_extract_result["kpts"],
            right_extract_result["kpts"],
            left_extract_result["descps"],
            right_extract_result["descps"],
            left_extract_result["mask"],
            right_extract_result["mask"],
            image_shape,
            image_shape,
            match_threshold))

    with Timer(text="[stereo] Elapsed time: {milliseconds:.0f} ms"):
        results = asyncio.run(stereo_engine.infer(dummy_left, dummy_right))
