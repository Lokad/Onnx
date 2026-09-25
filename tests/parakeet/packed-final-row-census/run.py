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
BASE=ROOT/'artifacts/parakeet-packed-final-row-census-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-packed-final-row-census-20260925'
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport;transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
CONTRACTS=ROOT/'artifacts/parakeet-packed-final-row-build-amd-20260925'
PRODUCT=CONTRACTS
CONSUMER=ROOT/'artifacts/parakeet-owned-packed-weight-census-amd-20260925'
REMOTE_CONSUMER='/dev/shm/lokad-parakeet-owned-packed-weight-census-20260925/runtime'
OWNERSHIP=ROOT/'artifacts/parakeet-weight-ownership-amd-v2-20260925'
REMOTE_PRODUCT='/dev/shm/lokad-parakeet-packed-final-row-build-20260925/runtime'


def references():
    closure=read(CONTRACTS/'closed.json');assert closure['passed']
    assert pin(CONTRACTS/'closed.json')['sha256']=='6295ad30835b7a2a1694b580a8e1447a6a0cdb28828a5960fa6e0e7c3628576f'
    assert closure['analysis']==pin(CONTRACTS/'analysis.json')
    for name,wanted in closure['files'].items():assert pin(CONTRACTS/name)==wanted,name
    contracts=read(CONTRACTS/'analysis.json');review=read(PRODUCT/'build-review.json')
    assert contracts['passed'] and not contracts['release_admitted'] and contracts['compiled_review']==pin(PRODUCT/'build-review.json')
    assert [(s['mode'],s['passed']) for s in contracts['suites']]==[('512',41),('256',41),('scalar',2)]
    assert review['passed'] and review['helper_matches_proof'] and review['product']==contracts['product']
    assert pin(PRODUCT/'build-review.json')['sha256']=='18dd2739caca08c7ed4368589b70942c40665dd49ebd0cb107757c0605f119d8'
    assert review['source']['sha256']=='e3ca64b50ea4a5dd276d90a19779d602f8ed78b6275b9e5448fe3b021bec5e1b'
    assert [len(r['differences']) for r in review['methods']]==[3,0]
    consumer=read(CONSUMER/'build-review.json');built=read(CONSUMER/'build-collected/built.json')
    assert pin(CONSUMER/'build-review.json')['sha256']=='f47f279457cb6ab0b4f6f0d42337a86e4a49238a21c7757e9e22f934e893b617'
    assert consumer['passed'] and consumer['built']==pin(CONSUMER/'build-collected/built.json')
    for name,wanted in built['runtime'].items():assert pin(CONSUMER/'build-collected/runtime'/name)==wanted,name
    assert pin(OWNERSHIP/'closed.json')['sha256']=='abe97a41fbb60648a6ee55968cd5e36c861ac908603336bd2f465379cbb1f3cd'
    prior.references()
    return contracts


def prepare():
    contracts=references();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';(bundle/'source').mkdir(parents=True)
    def put(name,content):
        p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(content if isinstance(content,bytes) else content.encode())
    put('source/Program.cs',(CONSUMER/'bundle/source/Program.cs').read_bytes())
    put('source/NpySupport.cs',(OWNERSHIP/'bundle/source/NpySupport.cs').read_bytes())
    put('source/global.json',(ROOT/'global.json').read_bytes())
    project=(OWNERSHIP/'bundle/source/WeightOwnershipProbe.csproj').read_bytes()
    put('source/OwnedWeightCensus.csproj',project)
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('remote.py',(TOOLS/'vm.py').read_bytes())
    put('plan.md',(ROOT/'.agent/m78-parakeet-packed-final-row-20260925.md').read_bytes())
    for name,source in [('contracts.json',CONTRACTS/'closed.json'),('compiled-review.json',PRODUCT/'build-review.json')]:put('evidence/'+name,source.read_bytes())
    old=read(OWNERSHIP/'bundle/spec.json')
    names=['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']
    runtime={n+'.dll':pin(PRODUCT/'build-collected/runtime'/(n+'.dll')) for n in names}
    assert all(runtime[n]==v for n,v in contracts['product'].items())
    external={n:v for n,v in old['external'].items() if not n.startswith(old['original_runtime']+'/')}
    external.update({REMOTE_PRODUCT+'/'+n:v for n,v in runtime.items()})
    consumer={n:pin(CONSUMER/'build-collected/runtime'/n) for n in ['OwnedWeightCensus.dll','OwnedWeightCensus.deps.json','OwnedWeightCensus.runtimeconfig.json']}
    sources={n:REMOTE_PRODUCT+'/'+n for n in runtime}
    sources.update({n:REMOTE_CONSUMER+'/'+n for n in consumer})
    external.update({REMOTE_CONSUMER+'/'+n:v for n,v in consumer.items()})

    weights=read(OWNERSHIP/'capture-collected/probe/weights.json');retained=read(OWNERSHIP/'capture-collected/probe/retained.json')
    assert len(weights)==96 and sum(w['Cached'] for w in weights)==9
    assert len(retained)==553 and sum(w['PackedHash'] is not None for w in retained)==37
    spec=dict(boot=old['boot'],external=external,product=contracts['product'],runtime_files=runtime,original_runtime=REMOTE_PRODUCT,
        contracts=pin(CONTRACTS/'closed.json'),compiled_review=pin(PRODUCT/'build-review.json'),ownership=pin(OWNERSHIP/'closed.json'),
        consumer_files=consumer,runtime_sources=sources,consumer_build_review=pin(CONSUMER/'build-review.json'),consumer_rebuilt=False,
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
