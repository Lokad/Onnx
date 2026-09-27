"""Bind the qualified candidate to the complete weight census and public request."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from source import changed

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-attention-owned-census-amd-20260928'
REMOTE='/dev/shm/lokad-attention-owned-census-20260928'
CONTRACTS=ROOT/'artifacts/parakeet-attention-owned-build-amd-20260928'
OLD=ROOT/'artifacts/parakeet-owned-packed-weight-scope-census-amd-20260925'
CONSUMER=ROOT/'artifacts/parakeet-owned-packed-weight-census-amd-20260925'
APP=ROOT/'artifacts/parakeet-pointwise-tail-app-amd-20260927'
REMOTE_APP='/dev/shm/lokad-pwt-app-20260927'
REMOTE_PRODUCT='/dev/shm/lokad-attention-owned-build-20260928/runtime'


def load(name,path):
    loader=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(loader);loader.loader.exec_module(value)
    return value


prior=load('attention_census_transport',TOOLS.parent/'weight-ownership-probe/run.py')
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def references():
    assert pin(CONTRACTS/'closed.json')['sha256']=='a293403a5dc3a7bdabb7fdc448470feea2b54dbf139d98b7b3f9ed0eef39854f'
    closed=read(CONTRACTS/'closed.json');assert closed['passed']
    for name,wanted in closed['files'].items(): assert pin(CONTRACTS/name)==wanted,name
    proof=read(CONTRACTS/'analysis.json')
    assert closed['analysis']==pin(CONTRACTS/'analysis.json') and proof['passed']
    assert [(r['mode'],r['passed'],r['skipped']) for r in proof['suites']]==[('normal',93,0),('256',93,0),('scalar',26,0)]
    assert proof['compiled_review']==pin(CONTRACTS/'build-review.json')
    assert read(CONTRACTS/'build-review.json')['arithmetic_leaves_unchanged']
    assert pin(OLD/'closed.json')['sha256']=='577a07a60d901cf57c4ba46d70b2c6fc4c02148ab67a1c28fd78d9b4fbcc3d08'
    old_closed=read(OLD/'closed.json')
    assert old_closed['passed'] and old_closed['files']['bundle/spec.json']==pin(OLD/'bundle/spec.json')
    old=read(OLD/'bundle/spec.json')
    assert old['files']['source/Program.cs']==pin(OLD/'bundle/source/Program.cs')==pin(CONSUMER/'bundle/source/Program.cs')
    assert old['files']['source/NpySupport.cs']==pin(OLD/'bundle/source/NpySupport.cs')
    assert old['files']['source/OwnedWeightCensus.csproj']==pin(OLD/'bundle/source/OwnedWeightCensus.csproj')
    assert pin(APP/'closed.json')['sha256']=='505da6ab8c9ce1d61b6f2b241f2bbf45d156bd8c621e40bb2caed4d6967281a3'
    app=read(APP/'closed.json')
    assert app['passed'] and app['files']['collected/manifests/current-parakeet.json']==pin(APP/'collected/manifests/current-parakeet.json')
    manifest=read(APP/'collected/manifests/current-parakeet.json')
    for row in manifest['models'].values(): assert old['external'][row['path']]=={k:row[k] for k in ['bytes','sha256']}
    return proof,old,manifest


def prepare():
    assert not BASE.exists()
    contracts,old,manifest=references()
    from metadata import derive
    metadata=derive(manifest,old)
    values=changed((OLD/'bundle/source/Program.cs').read_bytes())
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    assert (TOOLS/'review.py').is_file() and (TOOLS/'README.md').is_file()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    put('source/Program.cs',values)
    for name in ['NpySupport.cs','OwnedWeightCensus.csproj','global.json']:
        put('source/'+name,(OLD/'bundle/source'/name).read_bytes())
    put('remote.py',(TOOLS/'vm.py').read_bytes())
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('protocol.md',(TOOLS/'README.md').read_bytes())
    put('prospective-plan.md',(ROOT/'PLAN.md').read_bytes())
    put('weight-identities.json',json.dumps(metadata,indent=2).encode())
    runtime={n+'.dll':pin(CONTRACTS/'build-collected/runtime'/(n+'.dll')) for n in
        ['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']}
    assert all(runtime[n]==v for n,v in contracts['product'].items())
    case,=[r for r in manifest['cases'] if r['name']=='1188-133604-0002']
    assert case==old['case'] and case['samples']==287360
    pcm_path=REMOTE_APP+'/assets/'+case['pcm']['path']
    external={r['path']:{k:r[k] for k in ['bytes','sha256']} for r in manifest['models'].values()}
    external[pcm_path]={k:case['pcm'][k] for k in ['bytes','sha256']}
    external.update({REMOTE_PRODUCT+'/'+n:v for n,v in runtime.items()})
    spec=dict(boot=old['boot'],external=external,product=contracts['product'],runtime_files=runtime,original_runtime=REMOTE_PRODUCT,
        contracts=pin(CONTRACTS/'closed.json'),compiled_review=pin(CONTRACTS/'build-review.json'),original_census=pin(OLD/'closed.json'),
        original_consumer_source=pin(OLD/'bundle/source/Program.cs'),application=pin(APP/'closed.json'),
        weights=metadata['weights'],retained=old['retained'],case=case,pcm_path=pcm_path,model_directory=old['model_directory'],
        feed=old['feed'],build_limits=old['build_limits'],capture_limits=old['capture_limits'],
        minimum_free=old['minimum_free'],output_limit=old['output_limit'],release_admitted=False,diagnostic_only=True,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py',
        'ort-diagnosis-amd/run.py','feed-forward-cost-diagnostic/isolated_baseline.py']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},helpers={n:pin(TOOLS.parent/n) for n in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),weights=len(metadata['weights']))))


def prepared():
    references();value=read(BASE/'prepared.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items(): assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items(): assert pin(TOOLS.parent/name)==wanted,name
    spec=read(BASE/'bundle/spec.json')
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name
    assert (BASE/'bundle/source/Program.cs').read_bytes()==changed((OLD/'bundle/source/Program.cs').read_bytes())
    assert spec['weights']==read(BASE/'bundle/weight-identities.json')['weights']


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    else:
        prepared()
        if action=='stage':transport.stage()
        else:
            kind=sys.argv[2];assert kind in ['build','capture']
            if action=='launch':
                if kind=='capture':assert read(BASE/'build-review-transferred.json')['passed']
                transport.launch(kind)
            else:{'observe':prior.observe,'collect':prior.collect}[action](kind)
