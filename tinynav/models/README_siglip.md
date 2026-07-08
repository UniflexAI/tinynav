# SigLIP semantic retrieval assets

TinyNav uses TensorRT engines for runtime semantic retrieval. The Python runtime does not require
`torch`.

Expected files:

- `siglip_vit_b_16_webli_image_fp16.onnx`
- `siglip_vit_b_16_webli_text_fp16.onnx`
- `siglip_vit_b_16_webli_tokenizer.json`

Build TensorRT plans for the current machine:

```bash
cd tinynav/models
make siglip
```

Generated runtime files:

- `siglip_vit_b_16_webli_image_fp16_$(uname -m).plan`
- `siglip_vit_b_16_webli_text_fp16_$(uname -m).plan`

If the exported text ONNX uses different input names or context length, override the text shape
argument:

```bash
make siglip SIGLIP_TEXT_SHAPES="'input_ids':1x64"
```

The image ONNX should accept `pixel_values` in `1x3x224x224` NCHW format. The wrapper applies
SigLIP-style resize and normalization to `[-1, 1]`.
