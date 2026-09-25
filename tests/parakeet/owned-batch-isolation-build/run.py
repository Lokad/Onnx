"""Build and qualify the single packed-dispatch relocation on AMD."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('depthwise_transport',TOOLS.parent/'weight-ownership-probe/run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
BASE=ROOT/'artifacts/parakeet-owned-batch-isolation-build-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-owned-batch-isolation-build-20260925'
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport;transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
SOURCE=ROOT/'artifacts/parakeet-owned-batch-isolation-source-20260925'
BEFORE=ROOT/'artifacts/parakeet-direct-depthwise-build-v2-amd-20260925'
RUNTIME=BEFORE/'build-collected/runtime'
REMOTE_RUNTIME='/dev/shm/lokad-parakeet-direct-depthwise-build-v2-20260925/runtime'


def source_verified():
    value=read(SOURCE/'prepared.json');assert value['passed'] and not value['root_product_changed'] and not value['release_admitted']
    for name,wanted in value['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    for name,wanted in value['tools'].items():assert pin(TOOLS.parent/'owned-batch-isolation-source'/name)==wanted,name
    assert value['plan']==pin(SOURCE/'prospective-plan.md') and value['patch']==pin(SOURCE/'candidate.patch')
    assert value['changed_product_files']==['src/Lokad.Onnx/TensorOps.MatMul.cs']
    assert value['changed_methods']==['RunBatchedFloatMatMul','MatMulInto','MatMul'] and len(value['source'])==435
    assert value['added_product_files']==[] and value['source_reversible'] and value['shared_dispatcher_matches_release']
    assert value['diagnosis']==pin(ROOT/'tests/benchmarks/e5-direct-code-results/inspection-20260925.json')
    baseline=read(BEFORE/'build-review.json');assert baseline['passed']
    assert baseline['product']['Lokad.Onnx.dll']['sha256']=='40260aef7fd93c5153601ec104a87843a2c017a2720fd64c9e24e3460d455749'
    for name,wanted in baseline['product'].items():assert pin(RUNTIME/name)==wanted
    return value,baseline


def prepare():
    assert not BASE.exists();source,baseline=source_verified()
    assert (TOOLS/'review.py').exists() and (TOOLS/'README.md').exists()
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for name in source['source']:put('source/'+name,(SOURCE/'source'/name).read_bytes())
    put('bridge-source/Program.cs',(TOOLS/'Bridge.cs.txt').read_bytes())
    put('source/tests/Lokad.Onnx.Backend.Tests/OwnedPackedRuntimeIdentityTests.cs',(TOOLS.parent/'packed-final-row-build/RuntimeIdentityTests.cs.txt').read_bytes())
    put('bridge-source/Bridge.csproj',(TOOLS.parent/'selected-profile-build-amd/Bridge.csproj').read_bytes())
    put('bridge-source/global.json',(ROOT/'global.json').read_bytes())
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('remote.py',(TOOLS/'vm.py').read_bytes())
    put('plan.md',(SOURCE/'prospective-plan.md').read_bytes())
    release=ROOT/'artifacts/parakeet-packed-final-row-graphs-amd-20260925/collected/runtimes/current'
    release_remote='/dev/shm/lokad-parakeet-packed-final-row-graphs-20260925/runtimes/current'
    release_files={p.name:pin(p) for p in release.glob('*.dll')}
    assert release_files['Lokad.Onnx.dll']['sha256']=='f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    runtime={n+'.dll':pin(RUNTIME/(n+'.dll')) for n in ['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']}
    spec=dict(boot=1789634288.0,prior=REMOTE_RUNTIME,before_product=baseline['product'],source_prepared=pin(SOURCE/'prepared.json'),
        external={**{REMOTE_RUNTIME+'/'+n:w for n,w in runtime.items()},**{release_remote+'/'+n:w for n,w in release_files.items()}},release_runtime=release_remote,release_product=release_files['Lokad.Onnx.dll'],release_admitted=False,failed_graph_cases=['e5-8tok'],
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=300),
        minimum_free=1024**3,output_limit=512*1024**2,expected_tests=dict(normal=64,scalar=25),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py','ort-diagnosis-amd/run.py',
        'feed-forward-cost-diagnostic/isolated_baseline.py','owned-packed-weight-build/Bridge.cs.txt','packed-final-row-build/RuntimeIdentityTests.cs.txt','selected-profile-build-amd/Bridge.csproj','direct-depthwise-build-v2/review.py']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},
        helpers={(TOOLS.parent/n).relative_to(ROOT).as_posix():pin(TOOLS.parent/n) for n in helpers},baseline_review=pin(BEFORE/'build-review.json')))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'))))


def prepared():
    source_verified();value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    assert value['baseline_review']==pin(BEFORE/'build-review.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


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
