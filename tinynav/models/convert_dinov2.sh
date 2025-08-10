ONNX_MODEL=./dinov2_base_224x224_fp16.onnx
OUTPUT_MODEL=$(basename $ONNX_MODEL .onnx)_$(uname -m).plan
echo "Converting $ONNX_MODEL to $OUTPUT_MODEL"
trtexec --onnx=$ONNX_MODEL \
	--shapes='pixel_values':1x3x224x224 \
        --fp16 \
        --saveEngine=$OUTPUT_MODEL

