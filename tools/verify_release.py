"""Offline integrity, source syntax and documentation-link checks."""
from pathlib import Path
import ast, hashlib, json, re
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'archive/source-manifest.json').read_text())
for row in manifest['files']:
    p=R/row['path']
    assert p.stat().st_size==row['bytes'],p
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
files=list(R.rglob('*.py'))
for p in files:
    if '.git' not in p.parts:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
for p in R.rglob('*.md'):
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
        if not target.startswith(('http:','https:','#','mailto:')):
            assert (p.parent/target.split('#')[0]).exists(),(p,target)
assert not any((R/n).exists() for n in ('runs','artifacts','data','.env'))
print(f"PASS: {len(manifest['files'])} source hashes; {len(files)} Python files; local documentation links; private asset exclusions")
