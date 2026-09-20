from pathlib import Path
import hashlib,json

CORE='48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710'
ORIGIN='900157f96572a273e961ff7e0c09f4e4bd6fa8e6af74ffabaa9fb39a708d6483'
FILTER='*LayerNormFloatInto* *WideOutput*'
LIMITS=dict(seconds=180,rss=6*1024**3,available=2*1024**3)
REMOTE='/home/vermorel/Onnx/artifacts/e5-layernorm-amd-proof-20260920'

def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
def verify(base):
    value=read(base/'bundle.json');assert value['core_sha256']==CORE and value['origin_sha256']==ORIGIN and value['limits']==LIMITS
    for name,want in value['files'].items():assert pin(base/name)==want,name
    assert pin(base/'origin/closed.json')['sha256']==ORIGIN
    old=read(base/'origin/closed.json');origins=read(base/'origin/selected.json')
    for name,source in origins.items():assert pin(base/name)==old['files'][source],name
    return value
