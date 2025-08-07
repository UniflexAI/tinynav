import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
import torch
from codetiming import Timer
from functools import lru_cache
import platform
import pycuda.autoinit # noqa: F401
import asyncio

def padding(img, shape):
    # Create an array with target shape and uint8 type
    filled = np.empty(shape, dtype=np.uint8)

    v_shift = max(0, (shape[0] - img.shape[0]) // 2)
    h_shift = max(0, (shape[1] - img.shape[1]) // 2)

    # Fill central image area
    filled[v_shift:v_shift + img.shape[0], h_shift:h_shift + img.shape[1]] = img

    # Fill top area
    if v_shift > 0:
        filled[:v_shift, h_shift:h_shift + img.shape[1]] = img[0:1].repeat(v_shift, axis=0)

    # Fill bottom area
    if v_shift + img.shape[0] < shape[0]:
        filled[v_shift + img.shape[0]:, h_shift:h_shift + img.shape[1]] = img[-1:].repeat(shape[0] - v_shift - img.shape[0], axis=0)

    # Fill left area
    if h_shift > 0:
        filled[:, :h_shift] = filled[:, h_shift:h_shift + 1].repeat(h_shift, axis=1)

    # Fill right area
    if h_shift + img.shape[1] < shape[1]:
        filled[:, h_shift + img.shape[1]:] = filled[:, h_shift + img.shape[1] - 1:h_shift + img.shape[1]].repeat(shape[1] - h_shift - img.shape[1], axis=1)

    return filled
        
def unpadding(img, shape):
    v_shift = (img.shape[0]-shape[0]) // 2
    h_shift = (img.shape[1]-shape[1]) // 2
    return img[v_shift:v_shift + shape[0], h_shift:h_shift + shape[1]]

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

    def memorized_infer(self, input_image:np.ndarray, threshold = np.array([0.0005], dtype=np.float32)):
        input_bytes = input_image.tobytes()
        return self.infer_cached(input_bytes, input_image.shape, input_image.dtype.str, threshold.item())

    @lru_cache(maxsize=128)
    def infer_cached(self, input_bytes, shape, dtype_str, threshold):
        input_image = np.frombuffer(input_bytes, dtype=dtype_str).reshape(shape)
        return self.infer(input_image, np.array([threshold]))


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
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image.unsqueeze(0).numpy()
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

class IGEVTRT(TRTBase):
    def __init__(self, scale = 0.5, engine_path=f"/tinynav/tinynav/models/rt_igev4gru_256x448_fp16_{platform.machine()}.plan"):
        super().__init__(engine_path)
        # model input [1,3,H,W]
        self.input_shape = self.inputs[0]["shape"][2:]
        self.scale = scale
        self.raw_image_shape = None
        self.resize_image_shape = None

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
                    shape = inputs[0]["shape"]
            print(name, shape, dtype)
            size = trt.volume(shape)
            nbytes = trt.volume(shape) * dtype.itemsize
            device_mem = cuda.mem_alloc(nbytes)
            bindings.append(int(device_mem))
            host_mem = cuda.pagelocked_empty(size, dtype)

            if is_input:
                inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                outputs.append({"host": host_mem, "device": device_mem, "shape": shape, "name": name})
        return inputs, outputs, bindings, stream

    async def infer(self, left_img:np.ndarray, right_img:np.ndarray):
        if self.raw_image_shape is None:
            self.raw_image_shape = left_img.shape
            self.resize_image_shape = (int(left_img.shape[0] * self.scale), int(left_img.shape[1] * self.scale))
        
        left_input = self.preprocess(left_img)
        right_input = self.preprocess(right_img)

        np.copyto(self.inputs[0]["host"], left_input.ravel())
        np.copyto(self.inputs[1]["host"], right_input.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        event = cuda.Event()
        event.record(self.stream)
        while not event.query():
            await asyncio.sleep(0)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        results = {}
        for out in self.outputs:
            result = np.zeros_like(out["host"])
            np.copyto(result, out["host"])
            result = result.reshape(self.input_shape)
            results[out["name"]] = result
        results["disp"] = self.postprocess(results["disp"])
        return results

    def preprocess(self, img):
        # resize && padding to input_shape
        resize_img = cv2.resize(img, (self.resize_image_shape[1], self.resize_image_shape[0]))
        resize_img = padding(resize_img, (self.input_shape[0], self.input_shape[1]))
        if len(resize_img.shape) == 2:
            resize_img = cv2.cvtColor(resize_img, cv2.COLOR_GRAY2RGB)
        resize_img = np.transpose(resize_img, (2, 0, 1))
        return resize_img
    
    def postprocess(self, img):
        if img is None:
            return None
        img = unpadding(img, self.resize_image_shape)
        img = cv2.resize(img, (self.raw_image_shape[1], self.raw_image_shape[0])) / self.scale
        return img


if __name__ == "__main__":
    dinov2 = Dinov2TRT()
    superpoint = SuperPointTRT()
    light_glue = LightGlueTRT()
    igev = IGEVTRT()
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
        print(embedding.shape)

    with Timer(text="[superpoint] Elapsed time: {milliseconds:.0f} ms"):
        left_extract_result = superpoint.infer(dummy_left)
        right_extract_result = superpoint.infer(dummy_right)

    with Timer(text="[lightglue] Elapsed time: {milliseconds:.0f} ms"):
        match_result = light_glue.infer(
            left_extract_result["kpts"],
            right_extract_result["kpts"],
            left_extract_result["descps"],
            right_extract_result["descps"],
            image_shape,
            image_shape,
            match_threshold,
        )

    with Timer(text="[igev] Elapsed time: {milliseconds:.0f} ms"):
        igev.infer(dummy_left, dummy_right)
        results = igev.infer_sync()
