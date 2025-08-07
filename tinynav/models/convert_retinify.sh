ONNX_MODEL=./retinify_0_1_4_480x848.onnx
OUTPUT_MODEL=$(basename $ONNX_MODEL .onnx)_$(uname -m).plan
echo "Converting $ONNX_MODEL to $OUTPUT_MODEL"
trtexec --onnx=$ONNX_MODEL \
        --minShapes='left':1x480x848x1,'right':1x480x848x1 \
        --maxShapes='left':1x480x848x1,'right':1x480x848x1 \
        --optShapes='left':1x480x848x1,'right':1x480x848x1 \
        --fp16 \
        --saveEngine=$OUTPUT_MODEL
