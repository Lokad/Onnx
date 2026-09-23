"""Freeze actual current binaries and the existing native reference fixtures."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save,CASES

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/release-graph-baseline-amd-v2-20260923'
SCREEN=ROOT/'artifacts/pyannote-winograd-output-blocks-screen-amd-20260923'
PARAKEET=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
SHARED=ROOT/'artifacts/shared-regression-20260918/reference'
E5=ROOT/'artifacts/e5-randomized-processes-20260921/payload/inputs'
FAILED=ROOT/'artifacts/release-graph-baseline-amd-20260923'
APP=ROOT/'artifacts/pyannote-winograd-product-app-amd-20260923'

def previous_closed():
    proof=read(FAILED/'closed.json');assert not proof['passed'] and proof['retained_failure'] and proof['no_numerical_or_timing_jobs']
    for name,wanted in proof['files'].items():assert pin(FAILED/name)==wanted,name
    assert pin(SCREEN/'closed.json')['sha256']=='354305cd6ce20ac21f538f8c1426003eed2ff6594e30b9932b76dbda9f5f9545'
    assert read(SCREEN/'closed.json')['passed'] and not read(SCREEN/'closed.json')['admitted']
    assert pin(PARAKEET/'closed.json')['sha256']=='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3'
    for name,wanted in read(ROOT/'artifacts/pyannote-winograd-output-blocks-source-20260923/prepared.json')['before'].items():assert pin(ROOT/name)==wanted,name

def prepare():
    previous_closed();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['Program.cs','NpySupport.cs','ReleaseBenchmark.csproj']:copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','native.py']:copy(TOOLS/name,bundle/'tools'/name)
    for name in ['Lokad.Onnx.dll','Google.Protobuf.dll']:copy(APP/'collected/runtimes/candidate'/name,bundle/'product'/name)
    core=pin(bundle/'product/Lokad.Onnx.dll')['sha256'];assert core=='521bae1702849ca23dda586515e7cbabaac2d1eabdff04dc90a7ba76059e93fb'
    cases=[];models={}
    previous=read(PARAKEET/'payload.json')
    for key in CASES[:5]:
        fixture=read(E5/(key+'.json'));copy(E5/(key+'.json'),bundle/'evidence'/(key+'.json'))
        source=ROOT/'models/multilingual-e5-small/model.onnx';assert pin(source)['sha256']==fixture['model_sha256']
        remote='/home/vermorel/Onnx/models/multilingual-e5-small/model.onnx';models[remote]=pin(source)
        ref=E5/fixture['reference_file'];assert pin(ref)['sha256']==fixture['reference_sha256']
        target='reference/e5/'+ref.name;copy(ref,bundle/target)
        cases.append(dict(key=key,model=remote,inputs=[dict(name=n,shape=[1,fixture['shape'][1]],values=v) for n,v in fixture['inputs'].items()],outputs=[dict(name='last_hidden_state',shape=fixture['shape'],file=target)]))
    shared=read(SHARED/'manifest.json');assert shared['ort']=='1.29.0'
    copy(SHARED/'manifest.json',bundle/'evidence/shared-manifest.json')
    for m in shared['models']:
        for asset in m['assets']:
            wanted={k:asset[k] for k in ['bytes','sha256']}
            # The VM path is the qualified immutable path; local ResNet moved directories.
            remote='/home/vermorel/Onnx/'+asset['file'];models[remote]=wanted
        step=m['scenarios'][0]['steps'][0];assert not step['carry']
        entry=dict(key=m['key'],model='/home/vermorel/Onnx/'+m['model'],inputs=[],outputs=[])
        for kind in ['inputs','outputs']:
            for item in step[kind]:
                source=SHARED/item['file'];assert pin(source)['sha256']==item['sha256']
                target='reference/shared/'+item['file'];copy(source,bundle/target)
                entry[kind].append(dict(name=item['name'],shape=item['shape'],file=target))
        cases.append(entry)
    assert [c['key'] for c in cases]==CASES
    save(bundle/'cases.json',dict(core=core,cases=cases))
    copy(FAILED/'collected/collection.json',bundle/'evidence/previous-collection.json')
    copy(FAILED/'closed.json',bundle/'evidence/retained-build-failure.json')
    copy(FAILED/'payload.json',bundle/'evidence/previous-payload.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    external=dict(read(SCREEN/'payload.json')['external'])
    for name,wanted in previous['external'].items():
        if any(name.startswith(p+'/') for p in previous['python_paths']) or name.startswith('/usr/') or name.startswith('/lib/'):
            assert name not in external or external[name]==wanted;external[name]=wanted
    for name,wanted in models.items():
        assert name not in external or external[name]==wanted;external[name]=wanted
    stage=dict(passed=True,external=external,python_paths=previous['python_paths'],interpreter=previous['interpreter'],
        product={p.name:pin(p) for p in (bundle/'product').iterdir()},
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),cases=CASES)))

if __name__=='__main__':prepare()
