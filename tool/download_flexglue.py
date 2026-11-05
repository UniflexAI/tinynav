"""Download the optional official v0.1 ONNX models, verifying SHA256."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request

from tinynav.core.flexglue import MODEL_DIR, MODEL_FILES, MODEL_URL


def sha256(path):
    with Path(path).open('rb') as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(model_dir):
    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in MODEL_FILES.items():
        target = root / name
        if target.exists() and sha256(target) == expected:
            print(f'Verified {target}')
            continue
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as destination:
            temporary = Path(destination.name)
        try:
            print(f'Downloading {MODEL_URL + name}', flush=True)
            urllib.request.urlretrieve(MODEL_URL + name, temporary)
            if sha256(temporary) != expected:
                raise ValueError(f'SHA256 mismatch for {name}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Verified {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', type=Path, default=MODEL_DIR)
    download(parser.parse_args().model_dir)
