ONNX_MODEL=./lightglue_v1.onnx
OUTPUT_MODEL=$(basename $ONNX_MODEL .onnx)_$(uname -m).plan
echo "Converting $ONNX_MODEL to $OUTPUT_MODEL"
trtexec --onnx=$ONNX_MODEL \
        --inputIOFormats="fp32:chw","fp32:chw","uint8:chw","uint8:chw","int32:chw","int32:chw","fp32:chw" \
                --minShapes='kpts0':1x1x2,'kpts1':1x1x2,'desc0':1x1x256,'desc1':1x1x256,'img_shape0':1x2,'img_shape1':1x2 \
                --optShapes='kpts0':1x1535x2,'kpts1':1x1546x2,'desc0':1x1535x256,'desc1':1x1546x256,'img_shape0':1x2,'img_shape1':1x2 \
                --maxShapes='kpts0':1x2048x2,'kpts1':1x2048x2,'desc0':1x2048x256,'desc1':1x2048x256,'img_shape0':1x2,'img_shape1':1x2 \
        --precisionConstraints=obey \
        --layerPrecisions="/backbone/self_attn.0/inner_attn/Einsum:fp16","/backbone/self_attn.0/inner_attn_1/Einsum:fp16" \
        --fp16 \
        --saveEngine=$OUTPUT_MODEL
