import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
from codetiming import Timer
import platform
import pycuda.autoinit # noqa: F401
import asyncio
from async_lru import alru_cache

class OutputAllocator(trt.IOutputAllocator):
    def __init__(self):
        super().__init__()
        self.buffers = {}
        self.shapes = {}
        self.sizes = {}

    def AddItem(self, size, name, shape, ptr):
        self.buffers[name] = ptr
        self.sizes[name] = size
        self.shapes[name] = shape

    def reallocate_output(self, tensor_name, memory, size, alignment):
        if self.sizes[tensor_name] < size:
            ptr = cuda.mem_alloc(size)
            self.sizes[tensor_name] = size
            self.buffers[tensor_name] = ptr
            return ptr
        else:
            return self.buffers[tensor_name]

    def notify_shape(self, tensor_name, shape):
        self.shapes[tensor_name] = tuple(shape)


class TRTBase:
    def __init__(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.output_allocator = OutputAllocator()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            if any(value for value in shape if value < 0):
                if is_input:
                    assert self.engine.num_optimization_profiles > 0
                    profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                    assert len(profile_shape) == 3  # min,opt,max
                    # Set the *max* profile as binding shape
                    shape = profile_shape[2]
                else:
                    shape[1] = 2048

            size = trt.volume(shape)
            nbytes = trt.volume(shape) * dtype.itemsize
            host_mem = None
            device_mem = cuda.mem_alloc(nbytes)
            bindings.append(int(device_mem))

            if is_input:
                host_mem = cuda.pagelocked_empty(size, dtype)
                inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                self.output_allocator.AddItem(nbytes, name, shape, device_mem)
                self.context.set_output_allocator(name, self.output_allocator)
                outputs.append({"device": device_mem, "dtype": dtype, "name": name})

        return inputs, outputs, bindings, stream

class SuperPointTRT(TRTBase):
    def __init__(self, engine_path=f"/tinynav/tinynav/models/superpoint_240x424_fp16_{platform.machine()}.plan"):
        super().__init__(engine_path)
        # model input [1,H,W,1]
        self.input_shape = self.inputs[0]["shape"][1:3] # [H,W]

    # default threshold as 
    # https://github.com/cvg/LightGlue/blob/746fac2c042e05d1865315b1413419f1c1e7ba55/lightglue/superpoint.py#L111
    #
    async def infer(self, input_image:np.ndarray, threshold = np.array([0.0005], dtype=np.float32)):
        # resize to input_size
        scale = self.input_shape[0] / input_image.shape[0]
        image = cv2.resize(input_image, (self.input_shape[1], self.input_shape[0]))

        np.copyto(self.inputs[0]["host"], image.ravel())
        np.copyto(self.inputs[1]["host"], threshold.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        event = cuda.Event()
        event.record(self.stream)
        while not event.query():
            await asyncio.sleep(0)

        results = {}
        for out in self.outputs:
            out_shape = self.output_allocator.shapes[out["name"]]
            out_device = self.output_allocator.buffers[out["name"]]
            results[out["name"]] = np.empty(out_shape, dtype=out["dtype"])
            cuda.memcpy_dtoh_async(results[out["name"]], out_device, self.stream)
        self.stream.synchronize()

        results["kpts"][0] = (results["kpts"][0] + 0.5) / scale - 0.5
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
    async def infer(self, kpts0, kpts1, desc0, desc1, img_shape0, img_shape1, match_threshold = np.array([0.1])):
        np.copyto(self.inputs[0]["host"][: kpts0.size], kpts0.ravel())
        np.copyto(self.inputs[1]["host"][: kpts1.size], kpts1.ravel())
        np.copyto(self.inputs[2]["host"][: desc0.size], desc0.ravel())
        np.copyto(self.inputs[3]["host"][: desc1.size], desc1.ravel())
        np.copyto(self.inputs[4]["host"], img_shape0.ravel())
        np.copyto(self.inputs[5]["host"], img_shape1.ravel())
        np.copyto(self.inputs[6]["host"], match_threshold.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        self.context.set_input_shape("kpts0", kpts0.shape)
        self.context.set_input_shape("kpts1", kpts1.shape)
        self.context.set_input_shape("desc0", desc0.shape)
        self.context.set_input_shape("desc1", desc1.shape)

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        event = cuda.Event()
        event.record(self.stream)
        while not event.query():
            await asyncio.sleep(0)

        results = {}
        for out in self.outputs:
            out_shape = self.output_allocator.shapes[out["name"]]
            out_device = self.output_allocator.buffers[out["name"]]
            results[out["name"]] = np.empty(out_shape, dtype=out["dtype"])
            cuda.memcpy_dtoh_async(results[out["name"]], out_device, self.stream)
        self.stream.synchronize()
        return results


class Dinov2TRT(TRTBase):
    def __init__(self, engine_path=f"/tinynav/tinynav/models/dinov2_base_224x224_fp16_{platform.machine()}.plan"):
        super().__init__(engine_path)

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))

            size = trt.volume(shape)
            nbytes = trt.volume(shape) * dtype.itemsize
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(nbytes)
            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                outputs.append({"host": host_mem, "device": device_mem, "shape": shape, "dtype": dtype, "name": name})

        return inputs, outputs, bindings, stream

    def preprocess_image(self, image, target_size=224):
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = np.transpose(image, (2, 0, 1))  # C x H x W
        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = (image - mean) / std

        image = np.expand_dims(image, axis=0) # 1 x C x H x W
        return image

    def infer(self, image):
        np.copyto(self.inputs[0]["host"], image.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        results = {}
        for out in self.outputs:
            result = np.zeros_like(out["host"])
            np.copyto(result, out["host"])
            results[out["name"]] = result.reshape(out["shape"])
        return results

class StereoEngineTRT:
    def __init__(self, engine_file_path=f"/tinynav/tinynav/models/retinify_0_1_4_480x848_{platform.machine()}.plan"):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs, self.outputs = [], []
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            vol = int(np.prod(shape).item()) if -1 not in shape else 0
            host = cuda.pagelocked_empty(vol, dtype) if vol > 0 else None
            dev = cuda.mem_alloc(host.nbytes) if vol > 0 else None
            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs).append(
                {'name': name, 'mode': mode, 'shape': shape, 'dtype': dtype, 'host': host, 'device': dev}
            )

    async def infer(self, left_img, right_img):
        self.im_shape = (480, 848)

        # Prepare tensors as 1x480x848x1
        left_tensor = left_img.astype(np.float32)[None, :, :, None]
        right_tensor = right_img.astype(np.float32)[None, :, :, None]

        # Handle dynamic shapes
        for tensor in self.inputs:
            if -1 in tensor['shape']:
                shape = left_tensor.shape
                self.context.set_input_shape(tensor['name'], shape)
                tensor['shape'] = shape
                tensor['vol'] = int(np.prod(shape))
                tensor['host'] = cuda.pagelocked_empty(tensor['vol'], tensor['dtype'])
                tensor['device'] = cuda.mem_alloc(tensor['host'].nbytes)

        for inp in self.inputs:
            if 'left' in inp['name'].lower():
                inp['host'][:] = left_tensor.flatten()
            elif 'right' in inp['name'].lower():
                inp['host'][:] = right_tensor.flatten()

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        event = cuda.Event()
        event.record(self.stream)
        while not event.query():
            await asyncio.sleep(0)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()

        # left right consistency check
        out = self.outputs[0]  # Assuming the first output is the disparity map
        out_map = out['host'].reshape(out['shape'])[0, :, :, 0]
        yy, xx = np.meshgrid(np.arange(out_map.shape[0]), np.arange(out_map.shape[1]), indexing='ij')
        invalid = (xx - out_map) < 0
        out_map[invalid] = np.inf
        return out_map.astype(np.float32)

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
        embedding = dinov2.infer(image)["last_hidden_state"][:, 0, :]
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
            image_shape,
            image_shape,
            match_threshold))

    with Timer(text="[stereo] Elapsed time: {milliseconds:.0f} ms"):
        results = asyncio.run(stereo_engine.infer(dummy_left, dummy_right))
