"""Bind the existing eight-case graph protocol to the current padding pair."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
import sys
PARENT=Path(__file__).resolve().parent.parent/'pad-current-graphs-amd'
sys.path.insert(1,str(PARENT))
from protocol import pin,read,save,JOBS,LIMITS
from consumer_scope import verify_scope

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-pad-current-graphs-v2-amd-20260926'
FAILED=ROOT/'artifacts/parakeet-pad-current-graphs-amd-20260926'
GRAPH=ROOT/'artifacts/parakeet-owned-batch-isolation-graphs-amd-20260925'
SHORT=ROOT/'artifacts/e5-steady-short-amd-20260925'
PRODUCT=ROOT/'artifacts/parakeet-pad-current-models-amd-20260926'
PRIOR=dict(graph=GRAPH,short=SHORT,models=PRODUCT,
    app=ROOT/'artifacts/parakeet-pad-current-app-amd-20260926',
    shared=ROOT/'artifacts/parakeet-pad-current-shared-amd-20260926',
    pyannote=ROOT/'artifacts/parakeet-pad-current-pyannote-amd-20260926')
REMOTE=dict(graph='/dev/shm/lokad-parakeet-owned-batch-isolation-graphs-20260925',
    short='/dev/shm/lokad-e5-steady-short-20260925',models='/dev/shm/lokad-parakeet-pad-current-models-20260926',
    app='/dev/shm/lokad-parakeet-pad-current-app-20260926',shared='/dev/shm/lokad-parakeet-pad-current-shared-20260926',
    pyannote='/dev/shm/lokad-parakeet-pad-current-pyannote-20260926')
DIGESTS=dict(graph='def19d3f178cbb318bc11999b6e23dd18db40772d78949707155cf1f4c791638',
    short='b81128a6dbea610cf0571279e078debe3ff24c2b0f3c89bbea0e0c4ab675629a',
    models='194eb9a48b64df22d19b51adfc3ebe3accc56004544620123c128c2a76f066bf',
    app='2b40b54d8e94de7326ceec5228ad1529c7513964211e8b5dfbb2c921cd798151',
    shared='d1cb211096aa6a02ea70601b93cb615186c5d3b00472dced1c53283a1a8d3e3c',
    pyannote='5e2b54bb685d9e954d3840524c486b55a22aadc5a574b6cc189bedf5bf487435')


def validate(reports):
    assert all(r['passed'] for r in reports.values())
    pair=reports['models']['identities']
    assert pair['selected']['Lokad.Onnx.dll']['sha256']=='f3992f40d889a932cd0d15e0323564db801a308f4b24d1848731af30ba3c19f6'
    assert pair['candidate']['Lokad.Onnx.dll']['sha256']=='a74acb17524f23be13e81ade871b2b2ffea2afde5bcdd12e339b75e0197edf10'
    assert all(reports[k]['identities']==pair for k in ['shared','pyannote'])
    app=reports['app'];assert app['identities']==dict(current=pair['selected'],candidate=pair['candidate'])
    assert app['performance']['admitted']
    for key,count in [('controls',63),('gates',21)]:
        assert len(app['performance'][key])==count and all(r['passed'] for r in app['performance'][key])
    pyannote=reports['pyannote'];assert pyannote['identity_guards']['passed'] and pyannote['identity_guards']['probes']==4
    assert pyannote['results']['candidate']['complete_public_results_exact']
    assert [r['key'] for r in reports['graph']['performance'] if not r['qualified']]==['e5-8tok']
    assert reports['short']['performance']['qualified']
    return {role:{'Lokad.Onnx.dll':pair[label]['Lokad.Onnx.dll']} for role,label in [('current','selected'),('candidate','candidate')]}


def previous_closed():
    verify_scope();reports={}
    assert pin(FAILED/'closed.json')['sha256']=='5745067dbaa155e0b4c69b91bf33b24b173e2420e857a5be93f6480088d90a11'
    failed=read(FAILED/'closed.json');assert not failed['passed'] and failed['no_inference']
    for name,wanted in failed['files'].items():assert pin(FAILED/name)==wanted,name
    assert read(FAILED/'analysis.json')['worker_created'] is False
    for label,folder in PRIOR.items():
        assert pin(folder/'closed.json')['sha256']==DIGESTS[label]
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
        reports[label]=read(folder/'analysis.json')
    assert read(PRIOR['app']/'closed.json')['admitted'] and read(SHORT/'closed.json')['admitted']
    assert not read(GRAPH/'closed.json')['admitted']
    return validate(reports)


def prepare():
    assert not BASE.exists();products=previous_closed()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals=verify_scope()
    def copy(source,target):
        target=bundle/target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','checks_e5.py','short_checks.py',
                 'reuse.py','native.py','native-e5.py','native-short.py']:
        copy((TOOLS if name=='remote_prepare.py' else PARENT)/name,'tools/'+name)
    for name in ['README.md']:copy(TOOLS/name,name)
    shutil.copy2(ROOT/'PLAN.md',bundle/'prospective-plan.md')
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json','payload.json']:
            copy(folder/name,'evidence/prerequisites/'+label+'/'+name)
        copy(folder/'collected/collection.json','evidence/prerequisites/'+label+'/collection.json')
    for name in ['closed.json','analysis.json','payload.json']:copy(FAILED/name,'evidence/launch-failure/'+name)
    copy(FAILED/'collected/collection.json','evidence/launch-failure/collection.json')
    copy(PRODUCT/'collected/evidence/compatibility.json','evidence/product-compatibility.json')
    copy(GRAPH/'collected/cases.json','cases.json')
    graph=read(GRAPH/'payload.json');links={};collection=read(GRAPH/'collected/collection.json')
    for name,wanted in graph['files'].items():
        if name.startswith(('reference/','runtimes/','runtimes-e5/','evidence/warmed-consumer/','evidence/e5-consumer/')) or name in ['evidence/baseline/payload.json','source/global.json']:
            assert pin(GRAPH/'collected'/name)==wanted==collection['files'][name]
            links[name]=dict(source=REMOTE['graph']+'/'+name,identity=wanted)
    short_files=read(SHORT/'collected/collection.json')['files']
    for name,wanted in short_files.items():
        if name.startswith('runtimes/'):
            links[name.replace('runtimes/','runtimes-short/')]=dict(source=REMOTE['short']+'/'+name,identity=wanted)
    for name in ['instructions.json','review.json']:
        copy(SHORT/'collected/consumer-inventory'/name,'evidence/short-consumer/'+name)
    for prefix in ['runtimes','runtimes-e5','runtimes-short']:
        for role,label in [('current','selected'),('candidate','candidate')]:
            name=prefix+'/'+role+'/Lokad.Onnx.dll';original=PRODUCT/'collected/runtimes'/label/'Lokad.Onnx.dll'
            assert pin(original)==products[role]['Lokad.Onnx.dll']
            links[name]=dict(source=REMOTE['models']+'/runtimes/'+label+'/Lokad.Onnx.dll',identity=pin(original))
    old_built=read(GRAPH/'collected/built.json');short_built=read(SHORT/'collected/built.json')
    built=dict(old_built,short_consumer=short_built['consumer'],files={
        name:row['identity'] for name,row in links.items() if Path(name).name.startswith('ReleaseBenchmark.')})
    save(bundle/'built.json',built)
    external=dict(graph['external'])
    for name,wanted in read(SHORT/'payload.json')['external'].items():
        assert name not in external or external[name]==wanted;external[name]=wanted
    stage=dict(passed=True,links=links,products=products,prerequisites=REMOTE,
        previous_owner=read(FAILED/'collected/collection.json')['identities'][0],
        consumer=built['consumer'],e5_consumer=built['e5_consumer'],short_consumer=built['short_consumer'],
        previous_consumer=graph['previous_consumer'],external=external,interpreter=graph['interpreter'],python_paths=graph['python_paths'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for path in [*TOOLS.iterdir(),*PARENT.iterdir()]:
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),jobs=len(JOBS),limits=LIMITS)))


if __name__=='__main__':prepare()
