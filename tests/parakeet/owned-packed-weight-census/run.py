"""Freeze one actual-model census against the qualified isolated product."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('ownership_transport',TOOLS.parent/'weight-ownership-probe/run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-census-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-owned-packed-weight-census-20260925'
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport;transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
CONTRACTS=ROOT/'artifacts/parakeet-owned-packed-weight-tests-amd-20260925'
PRODUCT=ROOT/'artifacts/parakeet-owned-packed-weight-amd-20260925'
OWNERSHIP=ROOT/'artifacts/parakeet-weight-ownership-amd-v2-20260925'
REMOTE_PRODUCT='/dev/shm/lokad-parakeet-owned-packed-weight-20260925/runtime'


def references():
    for folder,digest in [(CONTRACTS,'3fc949cad141d2a1fef1e3913436920f87f3cecb92f1bce94892c1037bc080f4'),
        (OWNERSHIP,'abe97a41fbb60648a6ee55968cd5e36c861ac908603336bd2f465379cbb1f3cd')]:
        assert pin(folder/'closed.json')['sha256']==digest
        closure=read(folder/'closed.json');assert closure['passed']
        for name,wanted in closure['files'].items():assert pin(folder/name)==wanted,name
    review=read(PRODUCT/'build-review.json');contracts=read(CONTRACTS/'analysis.json')
    assert review['passed'] and contracts['passed'] and not contracts['release_admitted']
    assert contracts['original_compiled_review']==pin(PRODUCT/'build-review.json')
    assert contracts['product']==review['product']
    prior.references()
    return contracts


def prepare():
    contracts=references();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';(bundle/'source').mkdir(parents=True)
    def put(name,content):
        p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(content if isinstance(content,bytes) else content.encode())
    put('source/Program.cs',(TOOLS/'Program.cs.txt').read_bytes())
    put('source/NpySupport.cs',(OWNERSHIP/'bundle/source/NpySupport.cs').read_bytes())
    put('source/global.json',(ROOT/'global.json').read_bytes())
    project=(OWNERSHIP/'bundle/source/WeightOwnershipProbe.csproj').read_bytes()
    put('source/OwnedWeightCensus.csproj',project)
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('remote.py',(TOOLS/'vm.py').read_bytes())
    put('plan.md',(ROOT/'.agent/m76-parakeet-owned-packed-weights-20260925.md').read_bytes())
    for name,source in [('contracts.json',CONTRACTS/'closed.json'),('compiled-review.json',PRODUCT/'build-review.json')]:put('evidence/'+name,source.read_bytes())
    old=read(OWNERSHIP/'bundle/spec.json')
    names=['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']
    runtime={n+'.dll':pin(PRODUCT/'build-collected/runtime'/(n+'.dll')) for n in names}
    assert all(runtime[n]==v for n,v in contracts['product'].items())
    external={n:v for n,v in old['external'].items() if not n.startswith(old['original_runtime']+'/')}
    external.update({REMOTE_PRODUCT+'/'+n:v for n,v in runtime.items()})
    weights=read(OWNERSHIP/'capture-collected/probe/weights.json');retained=read(OWNERSHIP/'capture-collected/probe/retained.json')
    assert len(weights)==96 and sum(w['Cached'] for w in weights)==9
    assert len(retained)==553 and sum(w['PackedHash'] is not None for w in retained)==37
    spec=dict(boot=old['boot'],external=external,product=contracts['product'],runtime_files=runtime,original_runtime=REMOTE_PRODUCT,
        contracts=pin(CONTRACTS/'closed.json'),compiled_review=pin(PRODUCT/'build-review.json'),ownership=pin(OWNERSHIP/'closed.json'),
        failed_release_controls=contracts['failed_release_controls'],release_admitted=False,diagnostic_only=True,
        weights=weights,retained=retained,case=old['case'],pcm_path=old['pcm_path'],model_directory=old['model_directory'],feed=old['feed'],
        build_limits=old['build_limits'],capture_limits=old['capture_limits'],minimum_free=old['minimum_free'],output_limit=128*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py','ort-diagnosis-amd/run.py','feed-forward-cost-diagnostic/isolated_baseline.py']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},helpers={str(TOOLS.parent/n):pin(TOOLS.parent/n) for n in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),weights=len(weights))))


def prepared():
    references();value=read(BASE/'prepared.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items():assert pin(Path(name))==wanted,name
    for name,wanted in read(BASE/'bundle/spec.json')['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


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
