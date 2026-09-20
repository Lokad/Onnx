from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
QUALIFIED='4f10e8bc70627f33d00ffb06ddccf15f43464d2a'
PRIOR=ROOT/'artifacts/e5-layernorm-product-20260920'
RECEIPT='5afd88417ad896ef416df9ddfe32d8a3bb61c32332f2cd7010ffe57d1c66b2b8'
PATHS=['src','tests/Lokad.Onnx.Backend.Tests','tests/Lokad.Onnx.Tensors.Tests','global.json','Directory.Build.props','Directory.Build.targets','NuGet.Config','nuget.config','pack.cmd','eng/smoke-pack.ps1']
SETTINGS=[dict(name='default',fingerprint=False,wide=False),dict(name='fingerprint',fingerprint=True,wide=False),dict(name='wide',fingerprint=False,wide=True),dict(name='both',fingerprint=True,wide=True)]
def pin(path):
    path=Path(path)
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def psutil_module():
    sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    import psutil
    return psutil
def absent(identity):
    ps=psutil_module()
    try:return ps.Process(identity['pid']).create_time()!=identity['birth']
    except ps.NoSuchProcess:return True
