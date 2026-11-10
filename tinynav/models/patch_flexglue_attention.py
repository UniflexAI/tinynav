"""Run the FlexGlue matcher's attention cores (MatMul -> Softmax -> MatMul) in BF16; the rest stays FP16.

With --stronglyTyped, TensorRT fuses each FP16 attention over the 64x64 grid (4096 tokens) into one kernel that
sporadically overflows and emits NaN; the NaN reaches covisibility_score and every match is dropped.
BF16 keeps FP16 speed with FP32 range.

Usage: python3 patch_flexglue_attention.py <in.onnx> <out.onnx>
"""
import sys

import onnx
from onnx import TensorProto, helper

src, dst = sys.argv[1], sys.argv[2]
model = onnx.shape_inference.infer_shapes(onnx.load(src))
graph = model.graph
dtype = {v.name: v.type.tensor_type.elem_type for v in [*graph.value_info, *graph.input, *graph.output]}
producer = {o: n for n in graph.node for o in n.output}
consumers = {}
for n in graph.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)

# Nodes are keyed by their first output: names are not unique in the export.
cast_inputs, cast_output = {}, set()
for softmax in graph.node:
    if softmax.op_type != 'Softmax' or dtype.get(softmax.input[0]) != TensorProto.FLOAT16:
        continue
    scores, users = producer.get(softmax.input[0]), consumers.get(softmax.output[0], [])
    if scores is None or scores.op_type != 'MatMul' or len(users) != 1 or users[0].op_type != 'MatMul':
        continue
    cast_inputs.setdefault(scores.output[0], set()).update({0, 1})
    cast_inputs.setdefault(users[0].output[0], set()).add(1)
    cast_output.add(users[0].output[0])

bf16 = set()
nodes = []
for n in graph.node:
    key = n.output[0] if n.output else None
    if key in cast_inputs:
        inputs = list(n.input)
        for i in sorted(cast_inputs[key]):
            if inputs[i] not in bf16:
                cast = f'{inputs[i]}_bf16_for_{key}'
                nodes.append(helper.make_node('Cast', [inputs[i]], [cast], to=TensorProto.BFLOAT16, name=cast))
                inputs[i] = cast
        if key in cast_output:
            nodes.append(helper.make_node('MatMul', inputs, [f'{key}_bf16'], name=n.name))
            nodes.append(helper.make_node('Cast', [f'{key}_bf16'], [key], to=TensorProto.FLOAT16, name=f'{key}_fp16'))
        else:
            nodes.append(helper.make_node('MatMul', inputs, list(n.output), name=n.name))
            bf16.update(n.output)
    elif n.op_type == 'Softmax' and n.input[0] in bf16:
        nodes.append(n)
        bf16.update(n.output)
    else:
        nodes.append(n)
if not cast_output:
    raise RuntimeError(f'{src}: no FP16 attention block found')

del graph.node[:]
graph.node.extend(nodes)
value_info = [v for v in graph.value_info if v.name not in bf16]
del graph.value_info[:]
graph.value_info.extend(value_info)
onnx.checker.check_model(model)
onnx.save(model, dst)
print(f'{dst}: {len(cast_output)} attention blocks in BF16')
