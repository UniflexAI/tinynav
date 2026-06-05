# DINO Refinement Pipeline

This workflow improves TinyNav image retrieval by collecting reviewed retrieval pairs, training a DINOv2 LoRA adapter, exporting a merged model, and building a TensorRT engine for retrieval evaluation.

Run commands inside the devcontainer unless a step explicitly says otherwise.

## 1. Generate Retrieval Pairs

Use a query rosbag and an existing TinyNav map. For refined-model evaluation, pass the refined DINO engine and re-embed the map so query and map embeddings come from the same model.

```bash
python3 tool/retrieve_from_rosbag_map.py \
  --bag_path /mnt/nas/share-all/junlinp/rosbag/beijing/tinynav_debug_bags/debug_2026_06_04_20_54_21 \
  --map_path /mnt/nas/share-all/junlinp/rosbag/beijing/map \
  --topic /camera/camera/infra1/image_rect_raw \
  --topk 3 \
  --threshold 0.85 \
  --every_n 1 \
  --out_jsonl /tinynav/tinynav_temp/retrieval_debug_2026_06_04_20_54_21_refined_thr085.jsonl \
  --save_debug_dir /tinynav/tinynav_temp/retrieval_debug_2026_06_04_20_54_21_refined_thr085_images \
  --review_root /tinynav/tinynav_temp/retrieval_from_bag_review_20_54_21_map_thr085_refined \
  --dino_engine_path /mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/tensorrt/dinov2_refined_20_54_21_map_thr085_x86_64.plan \
  --reembed_map \
  --map_embedding_cache /mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/retrieval_eval/map_embeddings_refined.npz
```

For the first bootstrap run, omit `--dino_engine_path`, `--reembed_map`, and `--map_embedding_cache` to use the default DINO engine and the map's stored embeddings.

Outputs:

- `--out_jsonl`: one row per query frame with retrieved candidates and PnP status.
- `--save_debug_dir`: flat debug image export.
- `--review_root/session_YYYYmmdd_HHMMSS`: Web UI review session with `sample_*/sample.json`, `query.png`, `retrieved_*.png`, and optional `pnp_match.png`.

## 2. Review Pair Labels

Start the review UI on a free port:

```bash
python3 tool/review_retrieval_labels.py \
  --session_dir /tinynav/tinynav_temp/retrieval_from_bag_review_20_54_21_map_thr085_refined/session_YYYYmmdd_HHMMSS \
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

## 3. Merge Labels Into Dataset

Merge reviewed pair labels into a Hugging Face-format local dataset:

```bash
python3 scripts/sync_retrieval_hf_dataset.py export \
  --session_root /tinynav/tinynav_temp/retrieval_from_bag_review_20_54_21_map_thr085_refined \
  --out_dir /tinynav/datasets/tinynav-retrieval-reviewed \
  --split train
```

To merge with and upload to a Hugging Face dataset repo:

```bash
python3 scripts/sync_retrieval_hf_dataset.py merge_upload \
  --session_root /tinynav/tinynav_temp/retrieval_from_bag_review_20_54_21_map_thr085_refined \
  --repo_id Junlinp/tinynav-retrieval-reviewed-public \
  --split train \
  --work_dir /tinynav/tinynav_temp/hf_merge_work
```

If Hugging Face is rate-limited, keep the local dataset and copy it to NAS:

```bash
cp -r /tinynav/datasets/tinynav-retrieval-reviewed \
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

## 4. Train DINOv2 LoRA

Train from a local HF-format dataset:

```bash
python3 tool/train_dinov2_lora.py \
  --base_model facebook/dinov2-base \
  --train_pairs /mnt/nas/share-all/junlinp/tinynav_dino_dataset/tinynav-retrieval-reviewed/train.jsonl \
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

## 5. Merge LoRA

```bash
python3 scripts/merge_dinov2_lora.py \
  --base_model facebook/dinov2-base \
  --lora_path /mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/lora_ckpt \
  --out_dir /mnt/nas/share-all/junlinp/tinynav_dino_dataset/refine_runs/<run_id>/merged_model
```

Use the same `--base_model` used during training.

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

## Notes

- Always use `--reembed_map` when changing DINO engines. Comparing refined query embeddings to old map embeddings gives inconsistent similarity scores.
- Keep the map embedding cache tied to the exact TensorRT engine used to produce it.
- The retrieval threshold is usually `0.85` for the Beijing runs in this workflow.
- TensorRT engines are architecture-specific. Name files with the target architecture, for example `*_x86_64.plan`.
- Hugging Face upload/download can fail with `429 Too Many Requests`; keep a local/NAS dataset copy so the pipeline can continue offline.
