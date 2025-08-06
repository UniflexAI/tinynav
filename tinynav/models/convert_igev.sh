trtexec --onnx=rt_igev4gru_stereo.onnx \
                --minShapes='left':1x3x256x448,'right':1x3x256x448 \
                --maxShapes='left':1x3x256x448,'right':1x3x256x448 \
                --optShapes='left':1x3x256x448,'right':1x3x256x448 \
        --fp16 \
        --saveEngine=./rt_igev4gru_256x448_fp16_x86_64.plan
