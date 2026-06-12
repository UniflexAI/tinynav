# DINO Refinement Pipeline

This workflow improves TinyNav image retrieval by collecting reviewed retrieval pairs, training a DINOv2 LoRA adapter, exporting a merged model, and building a TensorRT engine for retrieval evaluation.

Run commands inside the devcontainer unless a step explicitly says otherwise.

## Pipeline Overview

The full loop is:

```text
query rosbag + TinyNav map
  -> retrieval pairs
  -> manual true/false/uncertain labels
  -> Hugging Face-format dataset
  -> DINOv2 LoRA training
  -> merged DINO model
  -> ONNX export
  -> TensorRT engine
  -> refined retrieval evaluation
```

Use this workflow when the current DINO retrieval engine returns plausible but incorrect map matches and you need to add reviewed positives/negatives back into the retrieval model.

## Prerequisites

- Run inside the TinyNav devcontainer.
- Mount NAS at `/mnt/nas` when using the verified Beijing dataset and refine outputs.
- Use a GPU container for DINO training, ONNX export checks, TensorRT engine build, and retrieval eval.
- Authenticate with Hugging Face only if you need `merge_upload`. The local export and NAS copy steps work offline.
- Keep the query rosbag image topic consistent with the map camera stream. The Beijing runs use `/camera/camera/infra1/image_rect_raw`.

## Current NAS Paths

These paths are the latest verified Beijing DINO refinement run. Example commands below still use `<run_id>` placeholders for future runs.

- Dataset root: `/mnt/nas/share-all/junlinp/tinynav_dino_dataset`
- Local container dataset copy: `/tinynav/datasets/tinynav-retrieval-reviewed-20_54_21-map-thr085`
- Reviewed dataset: `/mnt/nas/share-all/junlinp/tinynav_dino_dataset/tinynav-retrieval-reviewed-20_54_21-map-thr085`
- Refine run: `/mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/nas_refine_20_54_21_map_thr085_20260605_145628`
- TensorRT engine: `/mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/nas_refine_20_54_21_map_thr085_20260605_145628/tensorrt/dinov2_refined_20_54_21_map_thr085_x86_64.plan`
- Refined map embedding cache: `/mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/nas_refine_20_54_21_map_thr085_20260605_145628/retrieval_eval/map_embeddings_refined_20_54_21_map_thr085.npz`

## Current Hugging Face Dataset

The reviewed retrieval dataset has been uploaded to:

```text
Junlinp/tinynav-retrieval-reviewed-public
```

Dataset URL:

```text
https://huggingface.co/datasets/Junlinp/tinynav-retrieval-reviewed-public
```

Last confirmed upload: 2026-06-12.

Current verified dataset contents:

- Rows: 93
- Images: 186
- `true_match`: 27
- `false_match`: 66

## Common Run Parameters

- Query topic: `/camera/camera/infra1/image_rect_raw`
- Retrieval threshold: `0.85` for the Beijing runs in this workflow.
- Candidate count: `--topk 3`
- Frame sampling: `--every_n 1`
- Query bag used for the verified run: `/mnt/nas/share-all/junlinp/rosbag/beijing/tinynav_debug_bags/debug_2026_06_04_20_54_21`
- Map used for the verified run: `/mnt/nas/share-all/junlinp/rosbag/beijing/map`

## 1. Generate Retrieval Pairs

Use a query rosbag and an existing TinyNav map.

For the first bootstrap run, use the default DINO engine and the map's stored embeddings. Do not pass `--dino_engine_path`, `--reembed_map`, or `--map_embedding_cache`.

```bash
python3 tool/retrieve_from_rosbag_map.py \
  --bag_path /mnt/nas/share-all/junlinp/rosbag/beijing/tinynav_debug_bags/debug_2026_06_04_20_54_21 \
  --map_path /mnt/nas/share-all/junlinp/rosbag/beijing/map \
  --topic /camera/camera/infra1/image_rect_raw \
  --topk 3 \
  --threshold 0.85 \
  --every_n 1 \
  --out_jsonl /tinynav/tinynav_temp/retrieval_debug_2026_06_04_20_54_21_thr085.jsonl \
  --save_debug_dir /tinynav/tinynav_temp/retrieval_debug_2026_06_04_20_54_21_thr085_images \
  --review_root /tinynav/tinynav_temp/retrieval_from_bag_review_20_54_21_map_thr085
```

For refined-model evaluation, use the command in [8. Evaluate The Refined Engine](#8-evaluate-the-refined-engine). A refined query embedding must be compared against map embeddings produced by the same refined engine.

Outputs:

- `--out_jsonl`: one row per query frame with retrieved candidates and PnP status.
- `--save_debug_dir`: flat debug image export.
- `--review_root/session_YYYYmmdd_HHMMSS`: Web UI review session with `sample_*/sample.json`, `query.png`, `retrieved_*.png`, and optional `pnp_match.png`.

Verify before continuing:

- The JSONL file exists and has one line per processed query frame.
- The review session directory contains `sample_*` folders.
- Each useful sample has `query.png`, at least one `retrieved_*.png`, and `sample.json`.

## 2. Review Pair Labels

Start the review UI on a free port:

```bash
python3 tool/review_retrieval_labels.py \
  --session_dir /tinynav/tinynav_temp/retrieval_from_bag_review_20_54_21_map_thr085/session_YYYYmmdd_HHMMSS \
  --host 0.0.0.0 \
  --port 8053
```

Open:

```text
http://127.0.0.1:8053/
```

Use only the per-retrieved-image labels:

- `true_match`
- `false_match`
- `uncertain`

The training dataset only uses retrieved-pair labels stored in `review_matches`. Whole-sample labels are not part of the DINO pair-training data.

The UI includes play mode for fast manual inspection:

- `all samples`
- `with retrieved candidates`
- `PnP success`

Verify before continuing:

- Label useful pairs from the retrieved-image cards, not from the whole-sample card.
- Open a labeled `sample.json` and confirm labels are written under `review_matches`.
- Use `uncertain` for ambiguous pairs so they can be filtered or reviewed later.

## 3. Merge Labels Into Dataset

Merge reviewed pair labels into a Hugging Face-format local dataset:

```bash
python3 scripts/sync_retrieval_hf_dataset.py export \
  --session_root /tinynav/tinynav_temp/retrieval_from_bag_review_20_54_21_map_thr085 \
  --out_dir /tinynav/datasets/tinynav-retrieval-reviewed-20_54_21-map-thr085 \
  --split train
```

To merge with and upload to a Hugging Face dataset repo:

```bash
python3 scripts/sync_retrieval_hf_dataset.py merge_upload \
  --session_root /tinynav/tinynav_temp/retrieval_from_bag_review_20_54_21_map_thr085 \
  --repo_id Junlinp/tinynav-retrieval-reviewed-public \
  --split train \
  --work_dir /tinynav/tinynav_temp/hf_merge_work
```

If the local HF-format dataset is already exported and only needs to be uploaded:

```bash
python3 scripts/sync_retrieval_hf_dataset.py upload \
  --local_dir /tinynav/datasets/tinynav-retrieval-reviewed-20_54_21-map-thr085 \
  --repo_id Junlinp/tinynav-retrieval-reviewed-public
```

If Hugging Face is rate-limited, keep the local dataset and copy it to NAS:

```bash
cp -r /tinynav/datasets/tinynav-retrieval-reviewed-20_54_21-map-thr085 \
  /mnt/nas/share-all/junlinp/tinynav_dino_dataset/
```

Dataset layout:

```text
train.jsonl
images/*.png
```

Each JSONL row contains:

- `query_image`
- `retrieved_image`
- `label`
- `query_timestamp_ns`
- `reference_timestamp_ns`
- `session_id`
- `sample_id`
- `reviewed_at`
- `review_note`

Verify before continuing:

- The local dataset has `train.jsonl` and `images/*.png`.
- `train.jsonl` contains the expected reviewed labels.
- The NAS copy exists if training will run from NAS.
- The Hugging Face dataset upload is optional. Do not block training on upload failures if the local or NAS dataset is available.

## 4. Train DINOv2 LoRA

Train from a local HF-format dataset:

```bash
python3 tool/train_dinov2_lora.py \
  --base_model facebook/dinov2-base \
  --train_pairs /mnt/nas/share-all/junlinp/tinynav_dino_dataset/tinynav-retrieval-reviewed-20_54_21-map-thr085/train.jsonl \
  --output_dir /mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/lora_ckpt \
  --batch_size 2 \
  --num_epochs 1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-2 \
  --image_height 224 \
  --image_width 224 \
  --contrastive_margin 0.8
```

If Hugging Face model download is unavailable, use an existing local merged model as the base:

```bash
--base_model tinynav_temp/lora/hf_refine_20260528_contrastive/merged_model
```

The training script resolves relative dataset image paths against the JSONL directory and uses ImageNet normalization to match `Dinov2TRT` runtime preprocessing.

Verify before continuing:

- The training log reports both positive and negative pairs.
- The LoRA checkpoint directory exists under `refine_runs/<run_id>/lora_ckpt`.
- If the dataset is small, start with `--num_epochs 1` and inspect retrieval eval before increasing epochs.

## 5. Merge LoRA

```bash
python3 scripts/merge_dinov2_lora.py \
  --base_model facebook/dinov2-base \
  --lora_path /mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/lora_ckpt \
  --out_dir /mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/merged_model
```

Use the same `--base_model` used during training.

Verify before continuing:

- The merged model directory exists.
- The merged model can be loaded by the ONNX export script.

## 6. Export ONNX

```bash
python3 scripts/export_dinov2_lora_onnx.py \
  --model_path /mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/merged_model \
  --out_onnx /mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/onnx/dinov2_refined.onnx \
  --height 224 \
  --width 224 \
  --opset 17
```

Validate the ONNX file:

```bash
python3 - <<'PY'
import onnx
path = '/mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/onnx/dinov2_refined.onnx'
model = onnx.load(path)
onnx.checker.check_model(model)
print('onnx ok:', path)
PY
```

Verify before continuing:

- `onnx.checker.check_model` succeeds.
- The ONNX input shape is compatible with `pixel_values:1x3x224x224`.
- The output embedding width is `768` for `facebook/dinov2-base`.

## 7. Build TensorRT Engine

Run this inside the devcontainer where TensorRT is installed:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/onnx/dinov2_refined.onnx \
  --saveEngine=/mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/tensorrt/dinov2_refined_x86_64.plan \
  --shapes=pixel_values:1x3x224x224 \
  --fp16
```

Sanity-check with TinyNav runtime:

```bash
python3 - <<'PY'
import asyncio
import numpy as np
from tinynav.core.models_trt import Dinov2TRT

engine = '/mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/tensorrt/dinov2_refined_x86_64.plan'
model = Dinov2TRT(engine_path=engine)
embedding = asyncio.run(model.infer(np.zeros((480, 848), dtype=np.uint8)))
print(embedding.shape, embedding.dtype, float(np.linalg.norm(embedding)))
PY
```

Expected embedding shape is `(768,)` for DINOv2-base.

Verify before continuing:

- `trtexec` writes the `.plan` file.
- `Dinov2TRT` loads the engine without TensorRT binding errors.
- Runtime inference returns a `(768,)` embedding with a finite norm.

## 8. Evaluate The Refined Engine

Run retrieval again with the refined TensorRT engine and re-embed/cache map embeddings:

```bash
python3 tool/retrieve_from_rosbag_map.py \
  --bag_path /mnt/nas/share-all/junlinp/rosbag/beijing/tinynav_debug_bags/debug_2026_06_04_20_54_21 \
  --map_path /mnt/nas/share-all/junlinp/rosbag/beijing/map \
  --topic /camera/camera/infra1/image_rect_raw \
  --topk 3 \
  --threshold 0.85 \
  --every_n 1 \
  --out_jsonl /tinynav/tinynav_temp/retrieval_refined_eval.jsonl \
  --save_debug_dir /tinynav/tinynav_temp/retrieval_refined_eval_images \
  --review_root /tinynav/tinynav_temp/retrieval_refined_eval_review \
  --dino_engine_path /mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/tensorrt/dinov2_refined_x86_64.plan \
  --reembed_map \
  --map_embedding_cache /mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/retrieval_eval/map_embeddings_refined.npz
```

Then open the review UI on the generated session:

```bash
python3 tool/review_retrieval_labels.py \
  --session_dir /tinynav/tinynav_temp/retrieval_refined_eval_review/session_YYYYmmdd_HHMMSS \
  --host 0.0.0.0 \
  --port 8053
```

Verify the refined eval:

- The map embedding cache is written by the same TensorRT engine passed with `--dino_engine_path`.
- The retrieval JSONL contains rows with refined similarity scores.
- The review session can be opened in the Web UI for manual comparison.
- If similarity scores look inconsistent, delete the map embedding cache and rerun with `--reembed_map`.

## Notes

- Always use `--reembed_map` when changing DINO engines. Comparing refined query embeddings to old map embeddings gives inconsistent similarity scores.
- Keep the map embedding cache tied to the exact TensorRT engine used to produce it.
- The retrieval threshold is usually `0.85` for the Beijing runs in this workflow.
- TensorRT engines are architecture-specific. Name files with the target architecture, for example `*_x86_64.plan`.
- Hugging Face upload/download can fail with `429 Too Many Requests`; keep a local/NAS dataset copy so the pipeline can continue offline.
