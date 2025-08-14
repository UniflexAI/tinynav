ONNX_MODEL=./superpoint.onnx
OUTPUT_MODEL=$(basename $ONNX_MODEL .onnx)_$(uname -m).plan
echo "Converting $ONNX_MODEL to $OUTPUT_MODEL"
trtexec --onnx=$ONNX_MODEL \
        --minShapes='image':1x240x424x1,'keypoint_threshold':1x1 \
        --maxShapes='image':1x240x424x1,'keypoint_threshold':1x1 \
        --optShapes='image':1x240x424x1,'keypoint_threshold':1x1 \
        --fp16 \
        --saveEngine=$OUTPUT_MODEL