#!/usr/bin/env python3
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import snapshot_download


def load_yaml(path: str):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pairs(path: str):
    rows = []
    base_dir = os.path.dirname(os.path.abspath(path))
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                for key in ("query_image", "retrieved_image"):
                    value = row.get(key, "")
                    if value and not os.path.isabs(value):
                        row[key] = os.path.join(base_dir, value)
                rows.append(row)
    return rows


def load_pairs_from_hf(repo_id: str, split: str = "train", revision: str | None = None) -> list[dict]:
    # Dataset layout in hub repo:
    # <split>.jsonl with records using image paths relative to repo root.
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
    retrieved = meta.get("retrieved_timestamps_ns", [])
    for i, rts in enumerate(retrieved):
        if int(rts) == int(ts):
            p = os.path.join(sample_dir, f"retrieved_{i:03d}.png")
            return p if os.path.exists(p) else None
    return None


def preprocess(path: str, h: int, w: int):
    img = Image.open(path).convert("RGB").resize((w, h))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return torch.from_numpy(arr)


def resolve_pair_images(row: dict) -> tuple[str | None, str | None]:
    # New HF dataset format: explicit image paths in row.
    if "query_image" in row and "retrieved_image" in row:
        return row["query_image"], row["retrieved_image"]
    # Backward compatibility with legacy local jsonl format.
    q_dir = os.path.dirname(row["sample_json"])
    q_path = os.path.join(q_dir, "query.png")
    r_path = ts_to_image(row["sample_json"], row["reference_timestamp_ns"])
    return q_path, r_path


def batch_iter(rows, bs):
    for i in range(0, len(rows), bs):
        yield rows[i:i+bs]


def contrastive_pair_loss(sim: torch.Tensor, y: torch.Tensor, margin: float) -> torch.Tensor:
    # sim in [-1, 1], convert to distance in [0, 2]
    dist = 1.0 - sim
    pos = y * (dist ** 2)
    neg = (1.0 - y) * (torch.clamp(margin - dist, min=0.0) ** 2)
    return torch.mean(pos + neg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--base_model", default="facebook/dinov2-base")
    parser.add_argument("--train_pairs", default="")
    parser.add_argument("--hf_dataset_repo", default="")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--hf_revision", default="")
    parser.add_argument("--output_dir", default="tinynav_temp/lora/default_run/lora_ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--image_height", type=int, default=224)
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--contrastive_margin", type=float, default=0.8)
    args = parser.parse_args()
    cfg = {}
    if args.config:
        cfg = load_yaml(args.config)

    def get_cfg(name: str, default):
        cli_val = getattr(args, name)
        if cli_val != parser.get_default(name):
            return cli_val
        return cfg.get(name, default)

    set_seed(int(get_cfg("seed", 42)))

    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model

    base_model = get_cfg("base_model", "facebook/dinov2-base")
    model = AutoModel.from_pretrained(base_model)
    lora_cfg = LoraConfig(
        r=int(get_cfg("lora_r", 8)),
        lora_alpha=int(get_cfg("lora_alpha", 16)),
        lora_dropout=float(get_cfg("lora_dropout", 0.05)),
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    hf_dataset_repo = get_cfg("hf_dataset_repo", "")
    train_pairs = get_cfg("train_pairs", "")
    if hf_dataset_repo:
        train = load_pairs_from_hf(
            repo_id=hf_dataset_repo,
            split=str(get_cfg("hf_split", "train")),
            revision=get_cfg("hf_revision", "") or None,
        )
    else:
        if not train_pairs:
            raise ValueError("Provide --hf_dataset_repo or --train_pairs (or via --config).")
        train = load_pairs(train_pairs)
    lr = float(get_cfg("learning_rate", 1e-4))
    wd = float(get_cfg("weight_decay", 1e-2))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    h = int(get_cfg("image_height", 224))
    w = int(get_cfg("image_width", 224))
    bs = int(get_cfg("batch_size", 8))
    epochs = int(get_cfg("num_epochs", 1))

    margin = float(get_cfg("contrastive_margin", 0.8))

    for epoch in range(epochs):
        random.shuffle(train)
        epoch_loss = 0.0
        n = 0
        for chunk in batch_iter(train, bs):
            q_batch, r_batch, y_batch = [], [], []
            for row in chunk:
                q_path, r_path = resolve_pair_images(row)
                if not (os.path.exists(q_path) and r_path and os.path.exists(r_path)):
                    continue
                q_batch.append(preprocess(q_path, h, w))
                r_batch.append(preprocess(r_path, h, w))
                y_batch.append(1.0 if row["label"] == "true_match" else 0.0)
            if not q_batch:
                continue

            q = torch.stack(q_batch).to(device)
            r = torch.stack(r_batch).to(device)
            y = torch.tensor(y_batch, dtype=torch.float32, device=device)

            pair_out = model(pixel_values=torch.cat([q, r], dim=0)).last_hidden_state[:, 0]
            q_out, r_out = torch.chunk(pair_out, 2, dim=0)
            q_out = F.normalize(q_out, dim=1)
            r_out = F.normalize(r_out, dim=1)
            sim = torch.sum(q_out * r_out, dim=1)

            loss = contrastive_pair_loss(sim, y, margin=margin)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item())
            n += 1

        print(f"epoch={epoch+1}/{epochs} loss={(epoch_loss/max(1,n)):.6f}")

    out_dir = get_cfg("output_dir", "tinynav_temp/lora/default_run/lora_ckpt")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"saved lora checkpoint: {out_dir}")


if __name__ == "__main__":
    main()
