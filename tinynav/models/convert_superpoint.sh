ONNX_MODEL=./superpoint_240x424_fp16.onnx
OUTPUT_MODEL=$(basename $ONNX_MODEL .onnx)_$(uname -m).plan
echo "Converting $ONNX_MODEL to $OUTPUT_MODEL"
trtexec --onnx=$ONNX_MODEL \
        --fp16 \
        --saveEngine=$OUTPUT_MODEL
