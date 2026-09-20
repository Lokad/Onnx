from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[3]
CORE='7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9'
ORIGIN='7778c67fa57d567ffa7d779b0960b78027f57fe1dff6bd448f7eca1c74542899'
PROTOCOL='whisper-natural-layer20-cross-v1'
KINDS={'managed':['MM','MN'],'native':['NM','NN']}

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)
def verify(spec):
    assert spec['protocol']==PROTOCOL and len(spec['requests'])==8 and len(spec['outputs'])==12
    assert (spec['workers'],spec['calls'],spec['arrays'],spec['output_payload_bytes'])==(16,32,384,4423680000)
    assert spec['scaled_error_limit']==1e-4
    for name,want in spec['files'].items():assert pin(ROOT/name)==want,name
    for name,want in spec['native_runtime']['files'].items():assert pin(Path(name))==want,name
def schedule(spec):
    requests=spec['requests'];assert len(requests)==8
    assert [(r['selected_request'],r['features'],r['original_request']) for r in requests]==[(i,f,[0,10,9,20][i]) for i in range(4) for f in ['managed','native']]
    assert [r['name'] for r in requests[:2]]==[r['name'] for r in requests[6:]]
    return [dict(engine=e,request=i,name=r['name'],features=r['features'],selected_request=r['selected_request'],id=f"{e}-{i:02}-{r['name']}-{r['features']}") for e in KINDS for i,r in enumerate(requests)]
def header(result,job,manifest):
    assert result['complete'] is True and result['engine']==job['engine'] and result['request_index']==job['request']
    assert result['manifest_sha256']==manifest and result['flags']=={} and len(result['records'])==2
    assert [r['kind'] for r in result['records']]==KINDS[job['engine']]
    assert all(r['inputs_unchanged'] is True and r['held_outputs_unchanged'] is True and len(r['outputs'])==12 for r in result['records'])
