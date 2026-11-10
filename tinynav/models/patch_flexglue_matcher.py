"""Replace each last-axis CumSum in the FlexGlue matcher ONNX by a MatMul with a constant upper-triangular matrix.

TensorRT 10.3 (Jetson) miscomputes CumSum when fused with the ops that feed it, so the soft-argmax denominators
become zero and every match is dropped. The MatMul is mathematically identical.

Usage: python3 patch_flexglue_matcher.py <in.onnx> <out.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
model = onnx.shape_inference.infer_shapes(onnx.load(src))
graph = model.graph
value_infos = [*graph.value_info, *graph.input, *graph.output]
dtype = {v.name: v.type.tensor_type.elem_type for v in value_infos}
shape = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in value_infos}
constant = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
constant.update({n.output[0]: numpy_helper.to_array(a.t) for n in graph.node if n.op_type == 'Constant' for a in n.attribute if a.name == 'value'})

nodes, cumsums = [], 0
for n in graph.node:
    if n.op_type != 'CumSum':
        nodes.append(n)
        continue
    out = n.output[0]
    attrs = {a.name: a.i for a in n.attribute}
    length = shape.get(n.input[0], [0])[-1]
    axis = constant.get(n.input[1])
    if axis is None or int(axis) != -1 or attrs.get('exclusive') or attrs.get('reverse') or not length or dtype.get(n.input[0]) != TensorProto.FLOAT:
        raise RuntimeError(f'{src}: unsupported CumSum {out}')
    graph.initializer.append(numpy_helper.from_array(np.triu(np.ones((length, length), np.float32)), f'{out}_triu'))
    nodes.append(helper.make_node('MatMul', [n.input[0], f'{out}_triu'], [out], name=n.name or out))
    cumsums += 1
if not cumsums:
    raise RuntimeError(f'{src}: no CumSum found')

del graph.node[:]
graph.node.extend(nodes)
onnx.checker.check_model(model)
onnx.save(model, dst)
print(f'{dst}: {cumsums} CumSum as MatMul')
