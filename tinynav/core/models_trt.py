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

class TRTFusionModel():
    def __init__(self,
        superpoint_engine_path=f"/tinynav/tinynav/models/superpoint_240x424_fp16_{platform.machine()}.plan",
        lightglue_engine_path=f"/tinynav/tinynav/models/lightglue_fp16_{platform.machine()}.plan"):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(superpoint_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.superpoint_engine = runtime.deserialize_cuda_engine(f.read())
        with open(lightglue_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.lightglue_engine = runtime.deserialize_cuda_engine(f.read())
        _, self.stream = cudart.cudaStreamCreate()
        self.superpoint_context = self.superpoint_engine.create_execution_context()
        self.lightglue_context = self.lightglue_engine.create_execution_context()
        self.memory_addresses = self.allocate_buffers()
        with Timer(name="[capture_graph]", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            self.superpoint_graph_exec, self.lightglue_graph_exec = self.capture_graph()
        logging.info(f"load {superpoint_engine_path} and {lightglue_engine_path} done!")

    def allocate_buffers(self):
        memory_addresses = {
            "superpoint": {},
            "lightglue": {},
        }
        for i in range(self.superpoint_engine.num_io_tensors):
            name = self.superpoint_engine.get_tensor_name(i)
            shape = self.superpoint_context.get_tensor_shape(name)
            dtype = np.dtype(trt.nptype(self.superpoint_engine.get_tensor_dtype(name)))
            ctype_dtype = numpy_to_ctypes[dtype]
            _ = self.superpoint_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
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

            memory_addresses["superpoint"][name] = {
                "host": host_mem,
                "device": device_ptr,
                "shape": shape,
                "nbytes": nbytes,
            }
        for i in range(self.lightglue_engine.num_io_tensors):
            name = self.lightglue_engine.get_tensor_name(i)
            if name in ["kpts1", "desc1", "mask1"]:
                continue
            shape = self.lightglue_context.get_tensor_shape(name)
            dtype = np.dtype(trt.nptype(self.lightglue_engine.get_tensor_dtype(name)))
            ctype_dtype = numpy_to_ctypes[dtype]
            _ = self.lightglue_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            size = trt.volume(shape)
            nbytes = trt.volume(shape) * dtype.itemsize

            # print(f"lightglue: name: {name}, shape: {shape}, dtype: {dtype}, ctype_dtype: {ctype_dtype}, is_input: {is_input}, size: {size}, nbytes: {nbytes}")
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

            memory_addresses["lightglue"][name] = {
                "host": host_mem,
                "device": device_ptr,
                "shape": shape,
                "nbytes": nbytes,
            }
        memory_addresses["lightglue"]["kpts1"] = memory_addresses["superpoint"]["kpts"]
        memory_addresses["lightglue"]["desc1"] = memory_addresses["superpoint"]["descps"]
        memory_addresses["lightglue"]["mask1"] = memory_addresses["superpoint"]["mask"]
        return memory_addresses
    def capture_graph(self):
        cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        for i in range(self.superpoint_engine.num_io_tensors):
            name = self.superpoint_engine.get_tensor_name(i)
            self.superpoint_context.set_tensor_address(name, self.memory_addresses["superpoint"][name]["device"])
        self.superpoint_context.execute_async_v3(stream_handle=self.stream)
        _, superpoint_graph = cudart.cudaStreamEndCapture(self.stream)
        _, superpoint_graph_exec = cudart.cudaGraphInstantiate(superpoint_graph, 0)
        cudart.cudaStreamSynchronize(self.stream)
        cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        for i in range(self.lightglue_engine.num_io_tensors):
            name = self.lightglue_engine.get_tensor_name(i)
            self.lightglue_context.set_tensor_address(name, self.memory_addresses["lightglue"][name]["device"])
        self.lightglue_context.execute_async_v3(stream_handle=self.stream)
        _, lightglue_graph = cudart.cudaStreamEndCapture(self.stream)
        _, lightglue_graph_exec = cudart.cudaGraphInstantiate(lightglue_graph, 1)
        cudart.cudaStreamSynchronize(self.stream)
        return superpoint_graph_exec, lightglue_graph_exec

    async def run_graph(self):
        if "aarch64" not in platform.machine():
                superpoint_addresses = self.memory_addresses["superpoint"]
                for name in ["image", "keypoint_threshold"]:
                    cudart.cudaMemcpyAsync(superpoint_addresses[name]["device"], superpoint_addresses[name]["host"].ctypes.data,
                                    superpoint_addresses[name]["nbytes"],
                                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                    self.stream)
                lightglue_addresses = self.memory_addresses["lightglue"]
                for name in ["kpts0", "desc0", "mask0", "threshold", "img_shape0", "img_shape1"]:
                    cudart.cudaMemcpyAsync(lightglue_addresses[name]["device"], lightglue_addresses[name]["host"].ctypes.data,
                                    lightglue_addresses[name]["nbytes"],
                                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                    self.stream)
        cudart.cudaGraphLaunch(self.superpoint_graph_exec, self.stream)
        cudart.cudaGraphLaunch(self.lightglue_graph_exec, self.stream)
        if "aarch64" not in platform.machine():
            superpoint_addresses = self.memory_addresses["superpoint"]
            for name in ["kpts", "scores", "descps", "mask"]:
                cudart.cudaMemcpyAsync(superpoint_addresses[name]["host"].ctypes.data, superpoint_addresses[name]["device"],
                                    superpoint_addresses[name]["nbytes"],
                                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                    self.stream)
            lightglue_addresses = self.memory_addresses["lightglue"]
            for name in ["match_indices", "score"]:
                cudart.cudaMemcpyAsync(lightglue_addresses[name]["host"].ctypes.data, lightglue_addresses[name]["device"],
                                    lightglue_addresses[name]["nbytes"],
                                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                    self.stream)
        _, event = cudart.cudaEventCreate()
        cudart.cudaEventRecord(event, self.stream)
        while cudart.cudaEventQuery(event)[0] == cudart.cudaError_t.cudaErrorNotReady:
            await asyncio.sleep(0)

        results = {}
        superpoint_addresses = self.memory_addresses["superpoint"]
        for name in ["kpts", "scores", "descps", "mask"]:
            results[name] = superpoint_addresses[name]["host"].copy()
        lightglue_addresses = self.memory_addresses["lightglue"]
        for name in ["match_indices", "score"]:
            results[name] = lightglue_addresses[name]["host"].copy()
        return results

    async def infer(self, kpts0, desc0, mask0, image_1, img_shape0, img_shape1, superpoint_threshold = np.array([[0.0005]], dtype=np.float32), match_threshold = np.array([[0.1]])):
        input_shape = self.memory_addresses["superpoint"]["image"]["shape"][2:4]
        scale = input_shape[0] / image_1.shape[0]
        network_input_shape = self.memory_addresses["superpoint"]["image"]["shape"][2:4]
        image = cv2.resize(image_1, (network_input_shape[1], network_input_shape[0]))
        image = image[None, None, :, :]
        np.copyto(self.memory_addresses["superpoint"]["image"]["host"], image)
        np.copyto(self.memory_addresses["superpoint"]["keypoint_threshold"]["host"], superpoint_threshold)
        recovered_kpts0 = (kpts0 + 0.5) * scale - 0.5

        np.copyto(self.memory_addresses["lightglue"]["kpts0"]["host"], recovered_kpts0)
        np.copyto(self.memory_addresses["lightglue"]["desc0"]["host"], desc0)
        np.copyto(self.memory_addresses["lightglue"]["mask0"]["host"], mask0)
        np.copyto(self.memory_addresses["lightglue"]["threshold"]["host"], match_threshold)
        np.copyto(self.memory_addresses["lightglue"]["img_shape0"]["host"], input_shape)
        np.copyto(self.memory_addresses["lightglue"]["img_shape1"]["host"], input_shape)

        results = await self.run_graph()
        results["kpts"][0] = (results["kpts"][0] + 0.5) / scale - 0.5
        results["mask"] = results["mask"][:, :, None]
        return results


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
        self.real_infer_cnt = 0

    # default threshold as
    # https://github.com/cvg/LightGlue/blob/746fac2c042e05d1865315b1413419f1c1e7ba55/lightglue/superpoint.py#L111
    #
    @alru_cache_numpy(maxsize=128)
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

class LightGlueTRT(TRTBase):
    def __init__(self, engine_path=f"/tinynav/tinynav/models/lightglue_fp16_{platform.machine()}.plan"):
        super().__init__(engine_path)
        self.real_infer_cnt = 0

    # default threshold as
    # https://github.com/cvg/LightGlue/blob/746fac2c042e05d1865315b1413419f1c1e7ba55/lightglue/lightglue.py#L333
    #
    @alru_cache_numpy(maxsize=128)
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
    def __init__(self, engine_path=f"/tinynav/tinynav/models/retinify_0_1_5_480x848_{platform.machine()}.plan"):
        super().__init__(engine_path)

    async def infer(self, left_img, right_img, baseline, focal_length):
        left_tensor = left_img[None, None, :, :]
        right_tensor = right_img[None, None, :, :]

        np.copyto(self.inputs[0]["host"], left_tensor)
        np.copyto(self.inputs[1]["host"], right_tensor)
        np.copyto(self.inputs[2]["host"], baseline)
        np.copyto(self.inputs[3]["host"], focal_length)

        results = await self.run_graph()
        return results['disp'][0, 0, :, :], results['depth'][0, 0, :, :]

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

    with Timer(text="[dinov2] Elapsed time: {milliseconds:.0f} ms"):
        embedding = asyncio.run(dinov2.infer(dummy_left))

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
        baseline = np.array([[0.05]])
        focal_length = np.array([[323.0]])
        disp, depth = asyncio.run(stereo_engine.infer(dummy_left, dummy_right, baseline, focal_length))

    trt_fusion_model = TRTFusionModel()
    with Timer(text="[trt_fusion] Elapsed time: {milliseconds:.0f} ms"):
        results = asyncio.run(trt_fusion_model.infer(
            left_extract_result["kpts"],
            left_extract_result["descps"],
            left_extract_result["mask"],
            dummy_left,
            image_shape,
            image_shape,
            threshold,
            match_threshold))
    def hard_time_cpu():
        time.sleep(0.1)
        print("hard time cpu")
    with Timer(text="[stereo & fution] Elapsed time: {milliseconds:.0f} ms"):
        async def stereo_and_fusion():
            baseline = np.array([[0.05]])
            focal_length = np.array([[323.0]])
            stereo_task = asyncio.create_task(stereo_engine.infer(dummy_left, dummy_right, baseline, focal_length))
            fusion_task = asyncio.create_task(trt_fusion_model.infer(
                left_extract_result["kpts"],
                left_extract_result["descps"],
                left_extract_result["mask"],
                dummy_left,
                image_shape,
                image_shape,
                threshold,
                match_threshold))
            # Run hard_time_cpu concurrently with other async tasks
            cpu_task = asyncio.to_thread(hard_time_cpu)
            # Wait for all tasks concurrently using gather
            (disp, depth), results, _ = await asyncio.gather(
                stereo_task,
                fusion_task,
                cpu_task
            )
            return disp, depth, results

        disp, depth, results = asyncio.run(stereo_and_fusion())
