from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[3]
CORE='7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9'
ORIGIN='513f355ef7a36959c6c6454fbb65da33d3feb20585104267bf500113b1da6054'
TRACE='0f45e6c282ead0d447f313727318714d2e5ec5910858bbad6de93c75cc483a96'
PROTOCOL='whisper-selected-natural-trace-v1'
KINDS={'managed':['MM','MN'],'native':['NN','NM']}

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
def verify(spec):
    assert spec['protocol']==PROTOCOL and len(spec['requests'])==4 and len(spec['outputs'])==41
    for name,want in spec['files'].items():assert pin(ROOT/name)==want,name
    for name,want in spec['native_runtime']['files'].items():assert pin(Path(name))==want,name
def schedule(spec):
    assert [r['original_request'] for r in spec['requests']]==[0,10,9,20]
    assert spec['requests'][0]['name']==spec['requests'][3]['name']
    return [dict(engine=e,request=i,name=r['name'],id=f"{e}-{i:02}-{r['name']}") for e in KINDS for i,r in enumerate(spec['requests'])]
def header(result,job,manifest):
    assert result['complete'] is True and result['engine']==job['engine'] and result['request_index']==job['request']
    assert result['manifest_sha256']==manifest and result['flags']=={} and len(result['records'])==2
    assert [r['kind'] for r in result['records']]==KINDS[job['engine']]
    assert all(r['inputs_unchanged'] is True and r['held_outputs_unchanged'] is True and len(r['outputs'])==41 for r in result['records'])
