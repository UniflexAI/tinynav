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
SUPERPOINT_ONNX  := superpoint_240x424_fp16.onnx
LIGHTGLUE_ONNX   := lightglue_fp16.onnx
RETINIFY_ONNX    := retinify_0_1_5_480x848.onnx

DINO_ENGINE       := $(basename $(DINO_ONNX))_$(ARCH).plan
SUPERPOINT_ENGINE := $(basename $(SUPERPOINT_ONNX))_$(ARCH).plan
LIGHTGLUE_ENGINE  := $(basename $(LIGHTGLUE_ONNX))_$(ARCH).plan
RETINIFY_ENGINE   := $(basename $(RETINIFY_ONNX))_$(ARCH).plan

ENGINES := $(DINO_ENGINE) $(SUPERPOINT_ENGINE) $(LIGHTGLUE_ENGINE) $(RETINIFY_ENGINE)

.PHONY: all dinov2 superpoint lightglue retinify clean help

all: $(ENGINES)

dinov2: $(DINO_ENGINE)
superpoint: $(SUPERPOINT_ENGINE)
lightglue: $(LIGHTGLUE_ENGINE)
retinify: $(RETINIFY_ENGINE)

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

$(SUPERPOINT_ENGINE): $(SUPERPOINT_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
		--fp16

$(LIGHTGLUE_ENGINE): $(LIGHTGLUE_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
		--precisionConstraints=obey \
		--layerPrecisions="/backbone/self_attn.0/inner_attn/Einsum:fp16","/backbone/self_attn.0/inner_attn_1/Einsum:fp16" \
		--fp16

$(RETINIFY_ENGINE): $(RETINIFY_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
		--precisionConstraints=obey \
		--layerPrecisions='/cost_volume/Unsqueeze:fp16,/cost_volume/post_process/conv_block3/conv_block3.0/BatchNormalization:fp32,/cost_volume/post_process/norm/norm.0/BatchNormalization:fp32,/disparity_refinement/disp_block3/disp_block3.2/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block3/disp_block3.2/block/block.1/Relu:fp32,/disparity_refinement/disp_block5/disp_block5.9/bn/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block5/disp_block5.9/bn/block/block.1/Relu:fp32,/disparity_refinement/disp_block5/disp_block5.10/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block5/disp_block5.10/block/block.1/Relu:fp32,/disparity_refinement/upsample2_conv/upsample2_conv.0/BatchNormalization:fp32' \
		--fp16

clean:
	rm -f *_$(ARCH).plan

