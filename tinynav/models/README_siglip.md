# SigLIP semantic retrieval assets

TinyNav uses TensorRT engines for runtime semantic retrieval. The Python runtime does not require
`torch`.

Expected files:

- `siglip_vit_b_16_webli_image_fp16.onnx`
- `siglip_vit_b_16_webli_text_fp16.onnx`
- `siglip_vit_b_16_webli_tokenizer.json`

Build TensorRT plans for the current machine. The Immich ONNX files are static-shape models, so no
shape override is needed:

```bash
cd tinynav/models
make siglip
```

Generated runtime files:

- `siglip_vit_b_16_webli_image_fp16_$(uname -m).plan`
- `siglip_vit_b_16_webli_text_fp16_$(uname -m).plan`

If an exported ONNX is dynamic-shape, override the shape arguments:

```bash
make siglip SIGLIP_IMAGE_SHAPES="'pixel_values':1x3x224x224" SIGLIP_TEXT_SHAPES="'input_ids':1x64"
```

The image ONNX should accept a `1x3x224x224` NCHW tensor. The wrapper applies SigLIP-style resize
and normalization to `[-1, 1]`.
