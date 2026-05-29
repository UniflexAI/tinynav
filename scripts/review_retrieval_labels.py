#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from glob import glob

from flask import Flask, redirect, render_template_string, request, url_for

HTML = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Retrieval Label Review</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; background: #f7fafc; }
    .row { display: flex; gap: 16px; flex-wrap: wrap; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
    img { max-width: 320px; border: 1px solid #ccc; }
    .meta { font-family: monospace; font-size: 12px; white-space: pre-wrap; }
    .btns button { margin-right: 8px; }
    .retrieval-item { margin-bottom: 14px; padding: 10px; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }
  </style>
</head>
<body>
  <h2>Retrieval Label Review</h2>
  <p>Sample {{ idx + 1 }} / {{ total }} | {{ sample_id }}</p>
  <form method=\"post\" action=\"{{ url_for('review', idx=idx) }}\" class=\"card\">
    <div class=\"btns\">
      <button type=\"submit\" name=\"review_label\" value=\"true_match\">true_match</button>
      <button type=\"submit\" name=\"review_label\" value=\"false_match\">false_match</button>
      <button type=\"submit\" name=\"review_label\" value=\"uncertain\">uncertain</button>
      {% if idx > 0 %}<a style=\"margin-left: 12px;\" href=\"{{ url_for('index', idx=idx-1) }}\">Prev</a>{% endif %}
      {% if idx + 1 < total %}<a style=\"margin-left: 12px;\" href=\"{{ url_for('index', idx=idx+1) }}\">Next</a>{% endif %}
    </div>
    <p><input style=\"width: 80%;\" type=\"text\" name=\"review_note\" placeholder=\"optional note\" /></p>
  </form>
  <div class=\"row\">
    <div class=\"card\">
      <h3>Query</h3>
      {% if query_path %}<img src=\"{{ url_for('media', relpath=query_path) }}\" />{% else %}<p>No query image</p>{% endif %}
    </div>
    <div class=\"card\">
      <h3>PnP / Inliers</h3>
      <p>pnp_success={{ pnp_success }} | inliers={{ inlier_count }} | inlier_ratio={{ '%.4f'|format(inlier_ratio) }}</p>
      {% if pnp_match_path %}<img src=\"{{ url_for('media', relpath=pnp_match_path) }}\" />{% else %}<p>No PnP match image</p>{% endif %}
    </div>
    <div class=\"card\">
      <h3>Retrieved</h3>
      {% for item in retrieved_items %}
        <div class=\"retrieval-item\">
          <p><b>#{{ item.idx }}</b> ts={{ item.timestamp_ns }} sim={{ '%.4f'|format(item.similarity) }}</p>
          {% if item.path %}<img src=\"{{ url_for('media', relpath=item.path) }}\" />{% else %}<p>Missing image</p>{% endif %}
          <form method=\"post\" action=\"{{ url_for('review_retrieved', idx=idx, ridx=item.idx) }}\">
            <button type=\"submit\" name=\"label\" value=\"true_match\">true_match</button>
            <button type=\"submit\" name=\"label\" value=\"false_match\">false_match</button>
            <button type=\"submit\" name=\"label\" value=\"uncertain\">uncertain</button>
            <input style=\"width: 60%;\" type=\"text\" name=\"note\" placeholder=\"optional note for this image\" />
          </form>
          <p>current: <b>{{ item.review_label }}</b> {% if item.review_note %}| note: {{ item.review_note }}{% endif %}</p>
        </div>
      {% endfor %}
      {% if not retrieved_paths %}<p>No retrieved images</p>{% endif %}
    </div>
  </div>
  <div class=\"card\">
    <h3>Metadata</h3>
    <div class=\"meta\">{{ meta }}</div>
  </div>
</body>
</html>
"""


def load_index(session_dir: str):
    index_path = os.path.join(session_dir, "index.jsonl")
    if not os.path.exists(index_path):
        sample_jsons = sorted(glob(os.path.join(session_dir, "sample_*", "sample.json")))
        if sample_jsons:
            rows = []
            for sample_json in sample_jsons:
                sample_dir = os.path.dirname(sample_json)
                sample_id = os.path.basename(sample_dir)
                with open(sample_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                rows.append({
                    "sample_id": sample_id,
                    "query_timestamp_ns": int(meta.get("query_timestamp_ns", 0)),
                    "sample_json": sample_json,
                })
            rows.sort(key=lambda x: x["query_timestamp_ns"])
            with open(index_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(f"[info] rebuilt missing index.jsonl from sample directories: {index_path}")
            return rows
        raise FileNotFoundError(
            f"Missing index file: {index_path}. "
            "Expected a session directory containing index.jsonl or sample_*/sample.json."
        )
    rows = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_session_dir(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    if os.path.isfile(path):
        raise ValueError(f"Expected directory, got file: {path}")

    # Direct session path.
    if os.path.exists(os.path.join(path, "index.jsonl")) or glob(os.path.join(path, "sample_*", "sample.json")):
        return path

    # Parent directory containing session_* directories.
    session_dirs = sorted([p for p in glob(os.path.join(path, "session_*")) if os.path.isdir(p)])
    if session_dirs:
        latest = session_dirs[-1]
        print(f"[info] using latest session directory: {latest}")
        return latest

    # Parent directory containing timestamp-like directories.
    any_dirs = sorted([p for p in glob(os.path.join(path, "*")) if os.path.isdir(p)])
    candidates = []
    for d in any_dirs:
        if os.path.exists(os.path.join(d, "index.jsonl")) or glob(os.path.join(d, "sample_*", "sample.json")):
            candidates.append(d)
    if candidates:
        latest = sorted(candidates)[-1]
        print(f"[info] using detected session directory: {latest}")
        return latest

    raise FileNotFoundError(
        f"Could not resolve a valid session directory from: {path}. "
        "Pass a directory that contains index.jsonl or sample_*/sample.json."
    )


def make_app(session_dir: str):
    rows = load_index(session_dir)
    if not rows:
        raise RuntimeError(f"No samples found in {session_dir}")
    app = Flask(__name__)

    @app.route("/media/<path:relpath>")
    def media(relpath):
        from flask import send_file
        # Accept both absolute paths and container-relative paths like
        # "tinynav/..." produced by url_for path encoding.
        if os.path.isabs(relpath):
            abspath = relpath
        elif relpath.startswith("tinynav/"):
            abspath = os.path.join("/", relpath)
        else:
            abspath = os.path.abspath(relpath)
        return send_file(abspath)

    @app.route("/")
    def root():
        return redirect(url_for("index", idx=0))

    @app.route("/sample/<int:idx>")
    def index(idx: int):
        idx = max(0, min(idx, len(rows) - 1))
        row = rows[idx]
        sample_json = row["sample_json"]
        with open(sample_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        sample_dir = os.path.dirname(sample_json)
        query_path = os.path.join(sample_dir, "query.png")
        query_path = query_path if os.path.exists(query_path) else None
        pnp_match_path = os.path.join(sample_dir, "pnp_match.png")
        pnp_match_path = pnp_match_path if os.path.exists(pnp_match_path) else None
        retrieved_paths = []
        retrieved_items = []
        retrieved_ts = meta.get("retrieved_timestamps_ns", [])
        similarities = meta.get("similarities", [])
        review_matches = meta.get("review_matches", [])
        for name in sorted(os.listdir(sample_dir)):
            if name.startswith("retrieved_") and name.endswith(".png"):
                retrieved_paths.append(os.path.join(sample_dir, name))
        for i, p in enumerate(retrieved_paths):
            review_label = "unreviewed"
            review_note = ""
            if i < len(review_matches):
                review_label = review_matches[i].get("label", "unreviewed")
                review_note = review_matches[i].get("note", "")
            retrieved_items.append({
                "idx": i,
                "path": p,
                "timestamp_ns": int(retrieved_ts[i]) if i < len(retrieved_ts) else -1,
                "similarity": float(similarities[i]) if i < len(similarities) else 0.0,
                "review_label": review_label,
                "review_note": review_note,
            })
        return render_template_string(
            HTML,
            idx=idx,
            total=len(rows),
            sample_id=row.get("sample_id", ""),
            query_path=query_path,
            pnp_match_path=pnp_match_path,
            pnp_success=bool(meta.get("pnp_success", False)),
            inlier_count=int(meta.get("inlier_count", 0)),
            inlier_ratio=float(meta.get("inlier_ratio", 0.0)),
            retrieved_paths=retrieved_paths,
            retrieved_items=retrieved_items,
            meta=json.dumps(meta, ensure_ascii=True, indent=2),
        )

    @app.post("/review/<int:idx>")
    def review(idx: int):
        idx = max(0, min(idx, len(rows) - 1))
        row = rows[idx]
        sample_json = row["sample_json"]
        label = request.form.get("review_label", "uncertain")
        note = request.form.get("review_note", "")
        with open(sample_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["review_label"] = label
        meta["review_note"] = note
        meta["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
        with open(sample_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
        with open(os.path.join(session_dir, "reviews.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sample_id": row.get("sample_id"),
                "review_label": label,
                "review_note": note,
                "reviewed_at": meta["reviewed_at"],
            }, ensure_ascii=True) + "\n")
        next_idx = min(idx + 1, len(rows) - 1)
        return redirect(url_for("index", idx=next_idx))

    @app.post("/review_retrieved/<int:idx>/<int:ridx>")
    def review_retrieved(idx: int, ridx: int):
        idx = max(0, min(idx, len(rows) - 1))
        row = rows[idx]
        sample_json = row["sample_json"]
        label = request.form.get("label", "uncertain")
        note = request.form.get("note", "")
        now = datetime.now().isoformat(timespec="seconds")
        with open(sample_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        retrieved_ts = meta.get("retrieved_timestamps_ns", [])
        review_matches = meta.get("review_matches", [])
        while len(review_matches) < len(retrieved_ts):
            review_matches.append({"label": "unreviewed", "note": "", "reviewed_at": ""})
        if 0 <= ridx < len(review_matches):
            review_matches[ridx] = {"label": label, "note": note, "reviewed_at": now}
        meta["review_matches"] = review_matches

        with open(sample_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
        with open(os.path.join(session_dir, "reviews.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sample_id": row.get("sample_id"),
                "retrieved_index": ridx,
                "retrieved_timestamp_ns": int(retrieved_ts[ridx]) if 0 <= ridx < len(retrieved_ts) else -1,
                "review_label": label,
                "review_note": note,
                "reviewed_at": now,
            }, ensure_ascii=True) + "\n")
        return redirect(url_for("index", idx=idx))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_dir", required=True, help="retrieval_debug/session_* directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8051)
    args = parser.parse_args()

    session_dir = resolve_session_dir(args.session_dir)
    app = make_app(session_dir)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
