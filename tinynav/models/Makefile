# =========================
# Simple TensorRT Makefile
# =========================
# Usage:
#   make            # build all engines
#   make dinov2     # build one engine
#   make clean      # remove generated .plan

ARCH      := $(shell uname -m)
TRTEXEC   ?= trtexec
SIGLIP_IMAGE_SHAPES ?=
SIGLIP_TEXT_SHAPES ?=

DINO_ONNX        := dinov2_base_224x224_fp16.onnx
SUPERPOINT_ONNX  := superpoint_fp16_dynamic.onnx
LIGHTGLUE_ONNX   := lightglue_fp16.onnx
RETINIFY_ONNX    := retinify_0_1_5_dynamic.onnx
FOUNDATION_STEREO_ONNX := foundation_stereo_11-33-40_256x320_4.onnx
SIGLIP_IMAGE_ONNX := siglip_vit_b_16_webli_image_fp16.onnx
SIGLIP_TEXT_ONNX  := siglip_vit_b_16_webli_text_fp16.onnx


DINO_ENGINE       := $(basename $(DINO_ONNX))_$(ARCH).plan
LIGHTGLUE_ENGINE  := $(basename $(LIGHTGLUE_ONNX))_$(ARCH).plan

# One plan each per model:
# - SuperPoint: dynamic engine covering RealSense + Looper.
# - Retinify: one profile, min/max shapes for RealSense (480x848) and Looper (640x544).
SUPERPOINT_ENGINE := $(basename $(SUPERPOINT_ONNX))_$(ARCH).plan
RETINIFY_ENGINE   := $(basename $(RETINIFY_ONNX))_$(ARCH).plan
FOUNDATION_STEREO_ENGINE := $(basename $(FOUNDATION_STEREO_ONNX))_$(ARCH).plan
SIGLIP_IMAGE_ENGINE := $(basename $(SIGLIP_IMAGE_ONNX))_$(ARCH).plan
SIGLIP_TEXT_ENGINE  := $(basename $(SIGLIP_TEXT_ONNX))_$(ARCH).plan

ENGINES := $(DINO_ENGINE) $(LIGHTGLUE_ENGINE) $(SUPERPOINT_ENGINE) $(RETINIFY_ENGINE)

.PHONY: help all dinov2 superpoint lightglue retinify foundation_stereo siglip clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

all: $(ENGINES) ## Build all TensorRT engines

dinov2: $(DINO_ENGINE) ## Build DINOv2 engine
superpoint: $(SUPERPOINT_ENGINE) ## Build SuperPoint engine
lightglue: $(LIGHTGLUE_ENGINE) ## Build LightGlue engine
retinify: $(RETINIFY_ENGINE) ## Build Retinify engine
foundation_stereo: $(FOUNDATION_STEREO_ENGINE) ## Build FoundationStereo engine
siglip: $(SIGLIP_IMAGE_ENGINE) $(SIGLIP_TEXT_ENGINE) ## Build SigLIP image/text engines

$(DINO_ENGINE): $(DINO_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
        --shapes='pixel_values':1x3x224x224 --fp16


$(LIGHTGLUE_ENGINE): $(LIGHTGLUE_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
        --precisionConstraints=obey \
        --layerPrecisions="/backbone/self_attn.0/inner_attn/Einsum:fp16","/backbone/self_attn.0/inner_attn_1/Einsum:fp16" \
        --fp16

# SuperPoint: one plan, dynamic (RealSense 240x424 + Looper 320x272).
$(SUPERPOINT_ENGINE): $(SUPERPOINT_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
        --minShapes=image:1x1x240x272 \
        --optShapes=image:1x1x320x424 \
        --maxShapes=image:1x1x320x424 \
        --fp16

# Retinify: one plan, one profile; min/max shapes cover RealSense (480x848) and Looper (640x544).
$(RETINIFY_ENGINE): $(RETINIFY_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
        --minShapes=left:1x480x544x1,right:1x480x544x1 \
        --optShapes=left:1x640x544x1,right:1x640x544x1 \
        --maxShapes=left:1x640x848x1,right:1x640x848x1 \
        --precisionConstraints=obey \
        --layerPrecisions='/cost_volume/Unsqueeze:fp16,/cost_volume/post_process/conv_block3/conv_block3.0/BatchNormalization:fp32,/cost_volume/post_process/norm/norm.0/BatchNormalization:fp32,/disparity_refinement/disp_block3/disp_block3.2/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block3/disp_block3.2/block/block.1/Relu:fp32,/disparity_refinement/disp_block5/disp_block5.9/bn/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block5/disp_block5.9/bn/block/block.1/Relu:fp32,/disparity_refinement/disp_block5/disp_block5.10/block/block.0/BatchNormalization:fp32,/disparity_refinement/disp_block5/disp_block5.10/block/block.1/Relu:fp32,/disparity_refinement/upsample2_conv/upsample2_conv.0/BatchNormalization:fp32' \
        --fp16

# foundation_stereo.onnx export settings:
#   --valid_iters 4
#   --max_disp 192
$(FOUNDATION_STEREO_ENGINE): $(FOUNDATION_STEREO_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ --builderOptimizationLevel=1

$(SIGLIP_IMAGE_ENGINE): $(SIGLIP_IMAGE_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
        $(if $(SIGLIP_IMAGE_SHAPES),--shapes=$(SIGLIP_IMAGE_SHAPES),) --fp16

$(SIGLIP_TEXT_ENGINE): $(SIGLIP_TEXT_ONNX)
	$(TRTEXEC) --onnx=$< --saveEngine=$@ \
        $(if $(SIGLIP_TEXT_SHAPES),--shapes=$(SIGLIP_TEXT_SHAPES),) --fp16


clean: ## Remove generated TensorRT plans for this architecture
	rm -f *_$(ARCH).plan
