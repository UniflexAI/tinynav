"""Small synchronous TensorRT session for dynamic FlexGlue inputs."""
from types import SimpleNamespace
import numpy as np
import tensorrt as trt
from cuda import cudart as cu

def checked(result):
 err,*values=result
 if err != cu.cudaError_t.cudaSuccess:raise RuntimeError(str(err))
 return values[0] if len(values)==1 else values

class Session:
 def __init__(self,path):
  self.logger=trt.Logger(trt.Logger.WARNING);self.runtime=trt.Runtime(self.logger)
  with open(path,'rb') as file:
   self.engine=self.runtime.deserialize_cuda_engine(file.read())
  if self.engine is None:raise RuntimeError('Failed to deserialize engine')
  self.context=self.engine.create_execution_context();self.stream=checked(cu.cudaStreamCreate());self.buffers={}
  self.names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
 def get_inputs(self):
  return [SimpleNamespace(name=n,shape=tuple(self.engine.get_tensor_shape(n))) for n in self.names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT]
 def run(self,outputs,feeds):
  feeds={n:np.array(v,dtype=trt.nptype(self.engine.get_tensor_dtype(n)),copy=True,order='C') for n,v in feeds.items()}
  for n,a in feeds.items():
   if not self.context.set_input_shape(n,a.shape):raise ValueError((n,a.shape))
  result={}
  for n in self.names:
   shape=tuple(self.context.get_tensor_shape(n));dtype=np.dtype(trt.nptype(self.engine.get_tensor_dtype(n)))
   if any(d<0 for d in shape):raise ValueError((n,shape))
   size=max(1,int(np.prod(shape)))*dtype.itemsize
   if n not in self.buffers or self.buffers[n][1]<size:
    if n in self.buffers:checked(cu.cudaFree(self.buffers[n][0]))
    self.buffers[n]=(checked(cu.cudaMalloc(size)),size)
   ptr=self.buffers[n][0];self.context.set_tensor_address(n,ptr)
   if n in feeds:checked(cu.cudaMemcpyAsync(ptr,feeds[n].ctypes.data,feeds[n].nbytes,cu.cudaMemcpyKind.cudaMemcpyHostToDevice,self.stream))
   else:result[n]=np.empty(shape,dtype=dtype)
  if not self.context.execute_async_v3(self.stream):raise RuntimeError('TensorRT execution failed')
  for n,a in result.items():checked(cu.cudaMemcpyAsync(a.ctypes.data,self.buffers[n][0],a.nbytes,cu.cudaMemcpyKind.cudaMemcpyDeviceToHost,self.stream))
  checked(cu.cudaStreamSynchronize(self.stream))
  return [result[n] for n in outputs]
 def close(self):
  checked(cu.cudaStreamSynchronize(self.stream))
  for ptr,_ in self.buffers.values():checked(cu.cudaFree(ptr))
  self.buffers.clear();checked(cu.cudaStreamDestroy(self.stream))
