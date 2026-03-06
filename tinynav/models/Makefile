# =========================
# Simple TensorRT Makefile
# =========================
# Usage:
#   make            # build all engines
#   make dinov2     # build one engine
#   make clean      # remove generated .plan

ARCH      := $(shell uname -m)
TRTEXEC   ?= trtexec

DINO_ONNX        := dinov2_base_224x224_fp16.onnx
SUPERPOINT_ONNX  := superpoint_fp16_dynamic.onnx
LIGHTGLUE_ONNX   := lightglue_fp16.onnx
RETINIFY_ONNX    := retinify_0_1_5_dynamic.onnx


DINO_ENGINE       := $(basename $(DINO_ONNX))_$(ARCH).plan
LIGHTGLUE_ENGINE  := $(basename $(LIGHTGLUE_ONNX))_$(ARCH).plan

SUPERPOINT_REALSENSE_ENGINE := $(basename $(SUPERPOINT_ONNX))_240x424_$(ARCH).plan
RETINIFY_REALSENSE_ENGINE   := $(basename $(RETINIFY_ONNX))_480x848_$(ARCH).plan
SUPERPOINT_LOOPER_ENGINE := $(basename $(SUPERPOINT_ONNX))_320x272_$(ARCH).plan
RETINIFY_LOOPER_ENGINE := $(basename $(RETINIFY_ONNX))_640x544_$(ARCH).plan

ENGINES := $(DINO_ENGINE) $(LIGHTGLUE_ENGINE) $(SUPERPOINT_REALSENSE_ENGINE) $(RETINIFY_REALSENSE_ENGINE) $(SUPERPOINT_LOOPER_ENGINE) $(RETINIFY_LOOPER_ENGINE)

.PHONY: all dinov2 superpoint lightglue retinify clean help

all: $(ENGINES)

dinov2: $(DINO_ENGINE)
superpoint: $(SUPERPOINT_REALSENSE_ENGINE) $(SUPERPOINT_LOOPER_ENGINE)
lightglue: $(LIGHTGLUE_ENGINE)
retinify: $(RETINIFY_REALSENSE_ENGINE) $(RETINIFY_LOOPER_ENGINE)

help:
	@echo "Targets:"
	@echo "  make all          # Build all engines"
	@echo "  make dinov2       # Build DINOv2"
	@echo "  make superpoint   # Build SuperPoint"
	@echo "  make lightglue    # Build LightGlue"
	@echo "  make retinify     # Build Retinify"
	@echo "  make clean        # Remove *_$(ARCH).plan"

$(DINO_ENGINE): $(DINO_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
	--shapes='pixel_values':1x3x224x224 --fp16


$(LIGHTGLUE_ENGINE): $(LIGHTGLUE_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
        --precisionConstraints=obey \
        --layerPrecisions="/backbone/self_attn.0/inner_attn/Einsum:fp16","/backbone/self_attn.0/inner_attn_1/Einsum:fp16" \
        --fp16

$(SUPERPOINT_REALSENSE_ENGINE): $(SUPERPOINT_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
        --minShapes=image:1x1x240x424 \
        --optShapes=image:1x1x240x424 \
        --maxShapes=image:1x1x240x424 \
        --fp16

$(RETINIFY_REALSENSE_ENGINE): $(RETINIFY_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
        --minShapes=left:1x1x480x848,right:1x1x480x848 \
    --optShapes=left:1x1x480x848,right:1x1x480x848 \
    --maxShapes=left:1x1x480x848,right:1x1x480x848 \
        --precisionConstraints=obey \
        --layerPrecisions='/cost_volume/Unsqueeze:fp16,/cost_volume/post_process/conv_block3/conv_block3.0/BatchNormalization:fp32,/cost_volume/post_process/norm/norm.0/BatchNormalization:fp32,/disparity_refinement/disp_block3/disp_block3.2/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block3/disp_block3.2/block/block.1/Relu:fp32,/disparity_refinement/disp_block5/disp_block5.9/bn/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block5/disp_block5.9/bn/block/block.1/Relu:fp32,/disparity_refinement/disp_block5/disp_block5.10/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block5/disp_block5.10/block/block.1/Relu:fp32,/disparity_refinement/upsample2_conv/upsample2_conv.0/BatchNormalization:fp32' \
        --fp16


$(SUPERPOINT_LOOPER_ENGINE): $(SUPERPOINT_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
        --minShapes=image:1x1x320x272 \
        --optShapes=image:1x1x320x272 \
        --maxShapes=image:1x1x320x272 \
        --fp16

$(RETINIFY_LOOPER_ENGINE): $(RETINIFY_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
    --minShapes=left:1x1x640x544,right:1x1x640x544 \
    --optShapes=left:1x1x640x544,right:1x1x640x544 \
    --maxShapes=left:1x1x640x544,right:1x1x640x544 \
    --precisionConstraints=obey \
    --layerPrecisions='/cost_volume/Unsqueeze:fp16,/cost_volume/post_process/conv_block3/conv_block3.0/BatchNormalization:fp32,/cost_volume/post_process/norm/norm.0/BatchNormalization:fp32,/disparity_refinement/disp_block3/disp_block3.2/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block3/disp_block3.2/block/block.1/Relu:fp32,/disparity_refinement/disp_block5/disp_block5.9/bn/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block5/disp_block5.9/bn/block/block.1/Relu:fp32,/disparity_refinement/disp_block5/disp_block5.10/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block5/disp_block5.10/block/block.1/Relu:fp32,/disparity_refinement/upsample2_conv/upsample2_conv.0/BatchNormalization:fp32' \
    --fp16

clean:
	rm -f *_$(ARCH).plan

