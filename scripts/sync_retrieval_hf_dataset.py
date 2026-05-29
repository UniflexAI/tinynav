#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from glob import glob
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _collect_rows_from_sessions(session_root: str, split: str) -> list[dict]:
    sample_jsons = sorted(glob(os.path.join(session_root, "session_*", "sample_*", "sample.json")))
    rows = []
    for sample_json in sample_jsons:
        with open(sample_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        sample_dir = os.path.dirname(sample_json)
        query_path = os.path.join(sample_dir, "query.png")
        if not os.path.exists(query_path):
            continue
        retrieved_ts = [int(x) for x in meta.get("retrieved_timestamps_ns", [])]
        review_matches = meta.get("review_matches", [])
        for i, rm in enumerate(review_matches):
            label = rm.get("label", "unreviewed")
            if label not in ("true_match", "false_match"):
                continue
            if i >= len(retrieved_ts):
                continue
            retrieved_path = os.path.join(sample_dir, f"retrieved_{i:03d}.png")
            if not os.path.exists(retrieved_path):
                continue
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "query_timestamp_ns": int(meta.get("query_timestamp_ns", -1)),
                    "reference_timestamp_ns": int(retrieved_ts[i]),
                    "query_src": query_path,
                    "retrieved_src": retrieved_path,
                    "session_id": os.path.basename(os.path.dirname(sample_dir)),
                    "sample_id": os.path.basename(sample_dir),
                    "reviewed_at": rm.get("reviewed_at", ""),
                    "review_note": rm.get("note", ""),
                }
            )
    return rows


def export_dataset(session_root: str, out_dir: str, split: str) -> str:
    rows = _collect_rows_from_sessions(session_root, split=split)
    images_dir = os.path.join(out_dir, "images")
    _safe_mkdir(images_dir)
    out_rows = []
    for idx, row in enumerate(rows):
        q_name = f"{split}_{idx:08d}_query.png"
        r_name = f"{split}_{idx:08d}_retrieved.png"
        q_dst = os.path.join(images_dir, q_name)
        r_dst = os.path.join(images_dir, r_name)
        shutil.copy2(row["query_src"], q_dst)
        shutil.copy2(row["retrieved_src"], r_dst)
        out_rows.append(
            {
                "query_image": os.path.join("images", q_name),
                "retrieved_image": os.path.join("images", r_name),
                "label": row["label"],
                "query_timestamp_ns": row["query_timestamp_ns"],
                "reference_timestamp_ns": row["reference_timestamp_ns"],
                "session_id": row["session_id"],
                "sample_id": row["sample_id"],
                "reviewed_at": row["reviewed_at"],
                "review_note": row["review_note"],
            }
        )
    _safe_mkdir(out_dir)
    split_path = os.path.join(out_dir, f"{split}.jsonl")
    with open(split_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "# TinyNav Retrieval Dataset\n\n"
            "Generated from reviewed retrieval sessions.\n\n"
            "Files:\n"
            "- `train.jsonl` (or split-specific jsonl)\n"
            "- `images/*.png`\n\n"
            "Each row includes `query_image`, `retrieved_image`, and `label`.\n"
        )
    return split_path


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_key(row: dict) -> tuple:
    return (
        int(row.get("query_timestamp_ns", -1)),
        int(row.get("reference_timestamp_ns", -1)),
        str(row.get("label", "")),
        str(row.get("session_id", "")),
        str(row.get("sample_id", "")),
    )


def _rebuild_images_and_split(rows: list[dict], split: str, out_dir: str) -> str:
    _safe_mkdir(out_dir)
    images_dir = os.path.join(out_dir, "images")
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    _safe_mkdir(images_dir)

    merged_rows = []
    for idx, row in enumerate(rows):
        query_src = row["query_src"]
        retrieved_src = row["retrieved_src"]
        if not os.path.exists(query_src) or not os.path.exists(retrieved_src):
            continue

        q_name = f"{split}_{idx:08d}_query.png"
        r_name = f"{split}_{idx:08d}_retrieved.png"
        q_dst = os.path.join(images_dir, q_name)
        r_dst = os.path.join(images_dir, r_name)
        shutil.copy2(query_src, q_dst)
        shutil.copy2(retrieved_src, r_dst)
        merged_rows.append(
            {
                "query_image": os.path.join("images", q_name),
                "retrieved_image": os.path.join("images", r_name),
                "label": row["label"],
                "query_timestamp_ns": int(row["query_timestamp_ns"]),
                "reference_timestamp_ns": int(row["reference_timestamp_ns"]),
                "session_id": row["session_id"],
                "sample_id": row["sample_id"],
                "reviewed_at": row.get("reviewed_at", ""),
                "review_note": row.get("review_note", ""),
            }
        )

    split_path = os.path.join(out_dir, f"{split}.jsonl")
    with open(split_path, "w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "# TinyNav Retrieval Dataset\n\n"
            "Generated from reviewed retrieval sessions.\n\n"
            "Files:\n"
            "- `train.jsonl` (or split-specific jsonl)\n"
            "- `images/*.png`\n\n"
            "Each row includes `query_image`, `retrieved_image`, and `label`.\n"
        )
    return split_path


def merge_upload_dataset(session_root: str, repo_id: str, split: str, work_dir: str, private: bool) -> dict:
    work_dir = os.path.abspath(work_dir)
    export_dir = os.path.join(work_dir, "new_export")
    remote_dir = os.path.join(work_dir, "remote_snapshot")
    merged_dir = os.path.join(work_dir, "merged_upload")

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    _safe_mkdir(work_dir)

    export_dataset(session_root=session_root, out_dir=export_dir, split=split)
    new_rows = _load_jsonl(os.path.join(export_dir, f"{split}.jsonl"))
    new_images_root = os.path.join(export_dir, "images")

    download_dataset(repo_id=repo_id, out_dir=remote_dir, revision=None)
    existing_rows = _load_jsonl(os.path.join(remote_dir, f"{split}.jsonl"))
    remote_images_root = os.path.join(remote_dir, "images")

    merged_rows = []
    seen = set()

    for row in existing_rows:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        merged_rows.append(
            {
                "query_src": os.path.join(remote_dir, row["query_image"]),
                "retrieved_src": os.path.join(remote_dir, row["retrieved_image"]),
                "label": row["label"],
                "query_timestamp_ns": int(row["query_timestamp_ns"]),
                "reference_timestamp_ns": int(row["reference_timestamp_ns"]),
                "session_id": row.get("session_id", ""),
                "sample_id": row.get("sample_id", ""),
                "reviewed_at": row.get("reviewed_at", ""),
                "review_note": row.get("review_note", ""),
            }
        )

    added_count = 0
    for row in new_rows:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        added_count += 1
        merged_rows.append(
            {
                "query_src": os.path.join(export_dir, row["query_image"]),
                "retrieved_src": os.path.join(export_dir, row["retrieved_image"]),
                "label": row["label"],
                "query_timestamp_ns": int(row["query_timestamp_ns"]),
                "reference_timestamp_ns": int(row["reference_timestamp_ns"]),
                "session_id": row.get("session_id", ""),
                "sample_id": row.get("sample_id", ""),
                "reviewed_at": row.get("reviewed_at", ""),
                "review_note": row.get("review_note", ""),
            }
        )

    split_path = _rebuild_images_and_split(merged_rows, split=split, out_dir=merged_dir)

    upload_dataset(local_dir=merged_dir, repo_id=repo_id, private=private)
    return {
        "repo_id": repo_id,
        "split": split,
        "existing_count": len(existing_rows),
        "new_export_count": len(new_rows),
        "merged_count": len(_load_jsonl(split_path)),
        "added_count": added_count,
        "work_dir": work_dir,
        "new_images_root": new_images_root,
        "remote_images_root": remote_images_root,
    }


def upload_dataset(local_dir: str, repo_id: str, private: bool) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=local_dir,
        path_in_repo=".",
    )


def download_dataset(repo_id: str, out_dir: str, revision: str | None) -> str:
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export")
    p_export.add_argument("--session_root", required=True, help="e.g. tinynav_temp/retrieval_debug")
    p_export.add_argument("--out_dir", required=True, help="local hf-format dataset dir")
    p_export.add_argument("--split", default="train")

    p_upload = sub.add_parser("upload")
    p_upload.add_argument("--local_dir", required=True)
    p_upload.add_argument("--repo_id", required=True, help="hf dataset repo id, e.g. org/tinynav-retrieval")
    p_upload.add_argument("--private", action="store_true", default=False)

    p_pull = sub.add_parser("download")
    p_pull.add_argument("--repo_id", required=True)
    p_pull.add_argument("--out_dir", required=True)
    p_pull.add_argument("--revision", default="")

    p_merge_upload = sub.add_parser("merge_upload")
    p_merge_upload.add_argument("--session_root", required=True)
    p_merge_upload.add_argument("--repo_id", required=True)
    p_merge_upload.add_argument("--split", default="train")
    p_merge_upload.add_argument("--work_dir", default="tinynav_temp/hf_merge_work")
    p_merge_upload.add_argument("--private", action="store_true", default=False)

    args = parser.parse_args()

    if args.cmd == "export":
        split_path = export_dataset(args.session_root, args.out_dir, args.split)
        print(f"exported split: {split_path}")
    elif args.cmd == "upload":
        upload_dataset(args.local_dir, args.repo_id, args.private)
        print(f"uploaded dataset: {args.repo_id}")
    elif args.cmd == "download":
        path = download_dataset(args.repo_id, args.out_dir, args.revision or None)
        print(f"downloaded dataset to: {path}")
    elif args.cmd == "merge_upload":
        stats = merge_upload_dataset(
            session_root=args.session_root,
            repo_id=args.repo_id,
            split=args.split,
            work_dir=args.work_dir,
            private=args.private,
        )
        print(json.dumps(stats, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
