trtexec --onnx=./dinov2_base.onnx \
                --minShapes='pixel_values':1x3x224x224 \
                --optShapes='pixel_values':1x3x224x224 \
                --maxShapes='pixel_values':1x3x224x224 \
                --device=0 \
        --fp16 \
        --saveEngine=./dinov2_fp16_jetson.plan
