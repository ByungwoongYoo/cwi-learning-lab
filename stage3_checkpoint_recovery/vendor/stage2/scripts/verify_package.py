"""Check package contents against its self-excluding integrity manifest."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'SHA256SUMS.json').read_text())
for name,sha in m.items():
    p=root/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha: raise SystemExit(f'FAIL: {name}')
print(f'PASS: {len(m)} packaged files checked (manifest is not an authenticity signature).')
