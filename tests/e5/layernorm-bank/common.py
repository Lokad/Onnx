from pathlib import Path
import hashlib,json

CORE='48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710'
ORIGIN='96ec37f6a9355a86d429822605dcc568b65bfaef870d4f61384dcacbe188a630'
REMOTE='/home/vermorel/Onnx/artifacts/e5-layernorm-bank-20260920'
REMOTE_ORIGIN='/home/vermorel/Onnx/artifacts/e5-layernorm-amd-proof-20260920'
LIMITS=dict(seconds=600,rss=3*1024**3,available=2*1024**3,foreign=.02,steal=.005)
VARIANTS=['Product','CopyA','CopyB','Wide']
CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']
BANKS=[dict(name=n,source=n,width=384,bias=True,diagnostic=False,repeats=r,rows=s) for n,r,s in zip(CASES,[128,32,8,8,2],[8,30,128,128,512])]
BANKS += [dict(name=f'diagnostic-{width}-'+('bias' if bias else 'no-bias'),source='e5-30tok',width=width,bias=bias,diagnostic=True,repeats=32,rows=30) for width in [383,385] for bias in [True,False]]

def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)
def order(visit):
    value=[BANKS[(i+visit)%9]['name'] for i in range(9)]
    return value[::-1] if visit%2 else value
def verify(base,origin):
    bundle=read(base/'bundle.json')
    assert bundle['protocol']=='layernorm-complete-bank-v1' and bundle['limits']==LIMITS and bundle['banks']==BANKS
    assert bundle['core']==CORE and bundle['origin_sha256']==ORIGIN and pin(base/'origin-closed.json')['sha256']==ORIGIN
    old=read(base/'origin-closed.json')
    for name,want in bundle['files'].items():assert pin(base/name)==want,name
    for name,want in bundle['origin_files'].items():assert pin(origin/name)==want==old['files']['collected/'+name],name
    assert pin(base/'source/Kernels.cs')==bundle['origin_files']['source/Kernels.cs']
    assert read(base/'banks.json')==BANKS
    return bundle
