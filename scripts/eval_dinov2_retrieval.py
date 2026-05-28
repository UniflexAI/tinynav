#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import snapshot_download


def preprocess(path: str, h: int, w: int):
    img = Image.open(path).convert("RGB").resize((w, h))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def load_pairs(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_pairs_from_hf(repo_id: str, split: str = "train", revision: str | None = None) -> list[dict]:
    local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision)
    pairs_path = os.path.join(local_dir, f"{split}.jsonl")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing split file in dataset repo: {pairs_path}")
    rows = load_pairs(pairs_path)
    for row in rows:
        if "query_image" in row and not os.path.isabs(row["query_image"]):
            row["query_image"] = os.path.join(local_dir, row["query_image"])
        if "retrieved_image" in row and not os.path.isabs(row["retrieved_image"]):
            row["retrieved_image"] = os.path.join(local_dir, row["retrieved_image"])
    return rows


def ts_to_image(sample_json: str, ts: int) -> str | None:
    sample_dir = os.path.dirname(sample_json)
    with open(sample_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for i, rts in enumerate(meta.get("retrieved_timestamps_ns", [])):
        if int(rts) == int(ts):
            p = os.path.join(sample_dir, f"retrieved_{i:03d}.png")
            return p if os.path.exists(p) else None
    return None


def resolve_pair_images(row: dict) -> tuple[str | None, str | None]:
    if "query_image" in row and "retrieved_image" in row:
        return row["query_image"], row["retrieved_image"]
    sample_dir = os.path.dirname(row["sample_json"])
    q_path = os.path.join(sample_dir, "query.png")
    r_path = ts_to_image(row["sample_json"], row["reference_timestamp_ns"])
    return q_path, r_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pairs", default="")
    parser.add_argument("--hf_dataset_repo", default="")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--hf_revision", default="")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()

    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.hf_dataset_repo:
        rows = load_pairs_from_hf(
            repo_id=args.hf_dataset_repo,
            split=args.hf_split,
            revision=args.hf_revision or None,
        )
    else:
        if not args.pairs:
            raise ValueError("Provide --pairs or --hf_dataset_repo")
        rows = load_pairs(args.pairs)
    scores = []
    labels = []

    with torch.no_grad():
        for row in rows:
            q_path, r_path = resolve_pair_images(row)
            if not (os.path.exists(q_path) and r_path and os.path.exists(r_path)):
                continue
            q = preprocess(q_path, args.height, args.width).unsqueeze(0).to(device)
            r = preprocess(r_path, args.height, args.width).unsqueeze(0).to(device)

            qv = F.normalize(model(pixel_values=q).last_hidden_state[:, 0], dim=1)
            rv = F.normalize(model(pixel_values=r).last_hidden_state[:, 0], dim=1)
            sim = float(torch.sum(qv * rv, dim=1).item())
            scores.append(sim)
            labels.append(1 if row["label"] == "true_match" else 0)

    if not scores:
        print("no valid pairs")
        return

    pred = [1 if s >= 0.5 else 0 for s in scores]
    acc = sum(int(p == y) for p, y in zip(pred, labels)) / len(labels)
    print(f"pairs={len(labels)}")
    print(f"accuracy@0.5={acc:.4f}")
    print(f"mean_similarity={float(np.mean(scores)):.4f}")


if __name__ == "__main__":
    main()
