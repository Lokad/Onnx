"""Freeze actual root bytes only after M43's complete model and performance gates."""
import ast,importlib.util,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-first-use-kernels-root-amd-20260923'
APPLIED=ROOT/'artifacts/parakeet-first-use-kernels-root-integration-20260923'
SOURCE=ROOT/'artifacts/parakeet-first-use-kernels-source-20260923'
BUILD=ROOT/'artifacts/parakeet-first-use-kernels-build-amd-20260923'
SELECTED=ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
PRIOR=dict(build=BUILD,parakeet=ROOT/'artifacts/parakeet-first-use-kernels-app-amd-20260923',
    pyannote=ROOT/'artifacts/parakeet-first-use-kernels-pyannote-app-amd-20260923',
    graphs=ROOT/'artifacts/warmed-release-amd-v2-20260923',selected=SELECTED)
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('product_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)

def gates():
    assert pin(SOURCE/'prepared.json')['sha256']=='829e26d7acd55ccf969f4292949abc19385a014f562517344a3265d42a0f51c0'
    source=read(SOURCE/'prepared.json');assert source['passed'] and len(source['source'])==421
    for name,wanted in source['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    for label,folder in PRIOR.items():
        proof=read(folder/'closed.json');assert proof['passed']
        if label in ['parakeet','pyannote','graphs']:assert proof['admitted']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    graph=PRIOR['graphs'];assert pin(graph/'payload.json')['sha256']=='813e9d3df0924779e65a3ec272ffc3558e50c86b362c19e5dc5a0d2383266ff1'
    analysis=read(graph/'analysis.json');assert analysis['clocks']==37512 and analysis['measured']==8640
    assert analysis['consumer']['implementation_flags_equal'] and analysis['consumer']['branches_locals_exceptions_equal']
    assert read(graph/'closed.json')['all_controls_passed']
    assert pin(BUILD/'closed.json')['sha256']=='2fb4e3e587ab463a965d7cd4290ffe3f37674182bb47b7ee529d040825c3f243'
    assert pin(SELECTED/'closed.json')['sha256']=='62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'
    return source

def previous_closed():
    source=gates();applied=read(APPLIED/'applied.json')
    assert applied['passed'] and applied['source_files']==source['source'] and applied['prepared']==pin(SOURCE/'prepared.json')
    assert applied['prerequisites']=={label:pin(folder/'closed.json') for label,folder in PRIOR.items()}
    for name,wanted in source['source'].items():assert pin(ROOT/name)==wanted,name

def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    applied=read(APPLIED/'applied.json')
    for name in applied['source_files']:copy(ROOT/name,bundle/'source'/name)
    assert len(originals)==421
    for p in (SELECTED/'bundle/consumer').iterdir():
        if p.is_file():copy(p,bundle/'consumer'/p.name)
    for name in ['Bridge.dll','Bridge.deps.json','Bridge.runtimeconfig.json']:copy(BUILD/'collected/bridge'/name,bundle/'bridge'/name)
    copy(SELECTED/'bundle/evidence/tensor-source.tar',bundle/'evidence/tensor-source.tar')
    for name in ['backend','tensors']:copy(SELECTED/'collected'/(name+'-tests')/(name+'.trx'),bundle/'evidence'/('selected-'+name+'.trx'))
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(APPLIED/'applied.json',bundle/'evidence/root-applied.json')
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    stage=dict(passed=True,measured=read(BUILD/'analysis.json')['built'],root_integration=pin(APPLIED/'applied.json'),
        source_scope='421 actual integrated root files, identical to the admitted isolated candidate.',
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),source_files=421)))

if __name__=='__main__':prepare()
