from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3]
CORE='7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
def verify(spec):
    for name,want in spec['files'].items():assert pin(ROOT/name)==want,name
    for name,want in spec['native_runtime']['files'].items():assert pin(Path(name))==want,name
