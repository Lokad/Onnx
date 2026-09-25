"""Freeze the prepared M78 observer only after complete release qualification."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from prepare_source import ROOT,TOOLS,OUT as SOURCE,PHASE,pin,read

BASE=ROOT/'artifacts/parakeet-packed-final-row-profile-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-packed-final-row-profile-20260925'
APP=ROOT/'artifacts/parakeet-packed-final-row-release-app-amd-20260925'
REMOTE_APP='/dev/shm/lokad-parakeet-packed-final-row-release-app-20260925'
RELEASE=dict(root=ROOT/'artifacts/parakeet-packed-final-row-root-amd-20260925',
    app=APP,graphs=ROOT/'artifacts/parakeet-packed-final-row-graphs-amd-20260925',
    pyannote=ROOT/'artifacts/parakeet-packed-final-row-pyannote-app-amd-20260925')


def release_gates():
    receipt=read(SOURCE/'prepared.json')
    assert receipt['passed'] and not receipt['built']
    assert pin(SOURCE/'prepared.json')['sha256']=='ba592033b74d0a1b489d8d712cfc620334e67a25bba3bd2222139a3ce4edb6d0'
    for name,wanted in receipt['files'].items():assert pin(SOURCE/name)==wanted,name
    for name,wanted in receipt['inputs'].items():assert pin(ROOT/name)==wanted,name
    results={}
    for label,folder in RELEASE.items():
        proof=read(folder/'closed.json');value=read(folder/'analysis.json')
        assert proof['passed'] and value['passed']
        if label!='root':assert proof['admitted']
        if label=='graphs':assert proof['all_controls_passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
        results[label]=value
    product=receipt['product']
    root=results['root']
    assert root['measured']==product and root['root_source_verified']
    assert root['inventory']['implementation_flags_equal'] and root['inventory']['public_surface_equal']
    assert root['package']['passed'] and root['consumer']['passed']
    assert results['app']['identities']['candidate']==results['pyannote']['identities']['candidate']==product
    assert results['app']['identities']['current']==results['pyannote']['identities']['selected']
    assert read(RELEASE['graphs']/'payload.json')['products']['candidate']['Lokad.Onnx.dll']==product['Lokad.Onnx.dll']
    applied=read(RELEASE['root']/'bundle/evidence/root-applied.json')
    assert len(applied['source_files'])==433
    for name,wanted in applied['source_files'].items():assert pin(ROOT/name)==wanted,name
    return receipt


def prepare():
    from consumer_scope import verify_scope
    assert not BASE.exists();verify_scope();source=release_gates()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();inputs={}
    def copy(path,name):
        target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    for name in source['files']:copy(SOURCE/name,name)
    copy(SOURCE/'prepared.json','evidence/observer-source-prepared.json')
    copy(PHASE/'build-collected/inventory/instructions.json','evidence/original-observer-instructions.json')
    copy(APP/'collected/evidence/candidate-public.json','evidence/candidate-public.json')
    for label,folder in RELEASE.items():
        for name in ['closed.json','analysis.json']:copy(folder/name,'evidence/release/'+label+'/'+name)
    for name in ['remote.py','RUNBOOK.md']:copy(TOOLS/name,name)
    copy(ROOT/'PLAN.md','prospective-plan.md');inputs.pop('PLAN.md')
    manifest=read(APP/'collected/manifests/current-parakeet.json');external={}
    for value in manifest['models'].values():external[value['path']]={k:value[k] for k in ['bytes','sha256']}
    for value in [manifest['reference'],*[c['pcm'] for c in manifest['cases']]]:
        external[REMOTE_APP+'/assets/'+value['path']]={k:value[k] for k in ['bytes','sha256']}
    for name in ['manifests/current-parakeet.json','runtime/protocol.py','runtime/campaign_processes.py']:
        external[REMOTE_APP+'/'+name]=pin(APP/'collected'/name)
        inputs[(APP/'collected'/name).relative_to(ROOT).as_posix()]=pin(APP/'collected'/name)
    spec=dict(boot=1789634288.0,prior=REMOTE+'/runtime-base',app=REMOTE_APP,external=external,
        source_receipt=pin(SOURCE/'prepared.json'),release_closures={k:pin(v/'closed.json') for k,v in RELEASE.items()},
        original_consumer=source['consumer'],core=source['product']['Lokad.Onnx.dll'],data=source['product']['Lokad.Onnx.Data.dll'],
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    (bundle/'spec.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf8')
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'),str(path))
            inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    for folder,names in [('managed-phase-amd',['remote.py','audit.py','test_attribution.py']),('ort-diagnosis-amd',['run.py'])]:
        for name in names:
            path=TOOLS.parent/folder/name;inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    value=dict(passed=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),inputs=inputs)
    with (BASE/'prepared.json').open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=True,archive=value['archive'],spec=value['spec'],source=spec['source_receipt'])))


if __name__=='__main__':prepare()
