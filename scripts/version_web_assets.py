"""Give Flutter entry scripts content-addressed URLs after a web build."""
import hashlib
from pathlib import Path
import shutil
import re

root = Path(__file__).resolve().parents[1] / 'app/frontend/build/web'
main = root / 'main.dart.js'
version = hashlib.sha256(main.read_bytes()).hexdigest()[:12]
name = f'main.{version}.dart.js'
shutil.copyfile(main, root / name)
bootstrap = root / 'flutter_bootstrap.js'
source = bootstrap.read_text()
source, count = re.subn(r'"mainJsPath":"main(?:\.[a-f0-9]+)?\.dart\.js"',
                        f'"mainJsPath":"{name}"', source)
if count != 1:
    raise RuntimeError('Expected one Flutter mainJsPath in bootstrap')
bootstrap.write_text(source)
index = root / 'index.html'
index.write_text(re.sub(r'src="flutter_bootstrap\.js(?:\?v=[a-f0-9]+)?"',
                        f'src="flutter_bootstrap.js?v={version}"', index.read_text()))
print(f'Versioned frontend assets: {version}')
