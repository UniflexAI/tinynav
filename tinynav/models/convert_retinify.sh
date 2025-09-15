ONNX_MODEL=./retinify_0_1_5_480x848.onnx
OUTPUT_MODEL=$(basename $ONNX_MODEL .onnx)_$(uname -m).plan
echo "Converting $ONNX_MODEL to $OUTPUT_MODEL"
trtexec --onnx=$ONNX_MODEL \
        --precisionConstraints="obey" \
        --layerPrecisions='/cost_volume/Unsqueeze:fp16,/cost_volume/post_process/conv_block3/conv_block3.0/BatchNormalization:fp32,/cost_volume/post_process/norm/norm.0/BatchNormalization:fp32,/disparity_refinement/disp_block3/disp_block3.2/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block3/disp_block3.2/block/block.1/Relu:fp32,/disparity_refinement/disp_block5/disp_block5.9/bn/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block5/disp_block5.9/bn/block/block.1/Relu:fp32,/disparity_refinement/disp_block5/disp_block5.10/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block5/disp_block5.10/block/block.1/Relu:fp32,/disparity_refinement/upsample2_conv/upsample2_conv.0/BatchNormalization:fp32' \
        --verbose \
        --profilingVerbosity=detailed \
        --memPoolSize=workspace:2048 \
        --fp16 \
        --saveEngine=$OUTPUT_MODEL
