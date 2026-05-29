#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from huggingface_hub import HfApi, hf_hub_download


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, state: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_repo", default="Junlinp/tinynav-retrieval-reviewed-public")
    parser.add_argument("--model_repo", default="Junlinp/tinynav-dinov2-refined")
    parser.add_argument("--split", default="train")
    parser.add_argument("--state_path", default="tinynav_temp/hf_monitor_state.json")
    parser.add_argument("--work_dir", default="tinynav_temp/hf_monitor_runs")
    parser.add_argument("--force", action="store_true", default=False)

    parser.add_argument("--base_model", default="facebook/dinov2-base")
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
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    api = HfApi()
    state_path = os.path.abspath(args.state_path)
    work_dir = os.path.abspath(args.work_dir)
    ensure_dir(work_dir)

    state = load_state(state_path)
    run_started_at = utc_now_iso()

    info = api.dataset_info(repo_id=args.dataset_repo)
    dataset_revision = info.sha
    target_file = f"{args.split}.jsonl"
    dataset_files = {s.rfilename for s in info.siblings}
    if target_file not in dataset_files:
        raise FileNotFoundError(
            f"Missing {target_file} in dataset repo {args.dataset_repo} at revision {dataset_revision}"
        )

    local_train = hf_hub_download(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        filename=target_file,
        revision=dataset_revision,
    )
    train_hash = sha256_file(local_train)

    prev_revision = state.get("last_dataset_revision", "")
    prev_hash = state.get("last_train_jsonl_hash", "")
    changed = (dataset_revision != prev_revision) or (train_hash != prev_hash)

    if not changed and not args.force:
        summary = {
            "status": "no_update",
            "dataset_repo": args.dataset_repo,
            "model_repo": args.model_repo,
            "dataset_revision": dataset_revision,
            "train_jsonl_sha256": train_hash,
            "run_started_at": run_started_at,
            "run_finished_at": utc_now_iso(),
        }
        state.update(summary)
        save_state(state_path, state)
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(work_dir, f"run_{run_id}")
    lora_dir = os.path.join(run_root, "lora_ckpt")
    merged_dir = os.path.join(run_root, "merged_model")
    onnx_path = os.path.join(run_root, "onnx", "model.onnx")
    ensure_dir(run_root)

    env = os.environ.copy()
    py = sys.executable

    train_cmd = [
        py,
        "scripts/train_dinov2_lora.py",
        "--base_model",
        args.base_model,
        "--hf_dataset_repo",
        args.dataset_repo,
        "--hf_split",
        args.split,
        "--output_dir",
        lora_dir,
        "--seed",
        str(args.seed),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--image_height",
        str(args.image_height),
        "--image_width",
        str(args.image_width),
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--contrastive_margin",
        str(args.contrastive_margin),
    ]
    merge_cmd = [
        py,
        "scripts/merge_dinov2_lora.py",
        "--base_model",
        args.base_model,
        "--lora_path",
        lora_dir,
        "--out_dir",
        merged_dir,
    ]
    export_cmd = [
        py,
        "scripts/export_dinov2_lora_onnx.py",
        "--model_path",
        merged_dir,
        "--out_onnx",
        onnx_path,
        "--height",
        str(args.image_height),
        "--width",
        str(args.image_width),
        "--opset",
        str(args.opset),
    ]

    run_cmd(train_cmd, env=env)
    run_cmd(merge_cmd, env=env)
    run_cmd(export_cmd, env=env)

    api.create_repo(repo_id=args.model_repo, repo_type="model", exist_ok=True)
    latest_path = "onnx/latest/model.onnx"
    rev_path = f"onnx/by_dataset_rev/{dataset_revision}.onnx"
    api.upload_file(
        repo_id=args.model_repo,
        repo_type="model",
        path_or_fileobj=onnx_path,
        path_in_repo=latest_path,
        commit_message=f"Update ONNX from dataset revision {dataset_revision}",
    )
    api.upload_file(
        repo_id=args.model_repo,
        repo_type="model",
        path_or_fileobj=onnx_path,
        path_in_repo=rev_path,
        commit_message=f"Add ONNX for dataset revision {dataset_revision}",
    )

    summary = {
        "status": "updated",
        "dataset_repo": args.dataset_repo,
        "model_repo": args.model_repo,
        "dataset_revision": dataset_revision,
        "train_jsonl_sha256": train_hash,
        "uploaded_latest_path": latest_path,
        "uploaded_revision_path": rev_path,
        "run_started_at": run_started_at,
        "run_finished_at": utc_now_iso(),
        "run_root": run_root,
    }
    state.update(
        {
            "last_dataset_revision": dataset_revision,
            "last_train_jsonl_hash": train_hash,
            "last_published_model_revision": dataset_revision,
            "last_status": "updated",
            "last_run_finished_at": summary["run_finished_at"],
            "last_run_root": run_root,
        }
    )
    save_state(state_path, state)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
