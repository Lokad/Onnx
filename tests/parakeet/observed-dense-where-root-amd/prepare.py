"""Freeze actual root bytes after the candidate's complete model and performance gates."""
import ast,importlib.util,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
from consumer_scope import verify_scope
from source_scope import verify_source,root_files
from admission import product_identities
from correction import verify_incident,INCIDENT
from graph_prerequisite import verify_bundle

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-observed-dense-where-root-amd-20260924'
APPLIED=ROOT/'artifacts/parakeet-observed-dense-where-root-integration-20260924'
SOURCE=ROOT/'artifacts/parakeet-observed-dense-where-source-20260924'
BUILD=ROOT/'artifacts/parakeet-observed-dense-where-models-amd-20260924'
SELECTED=ROOT/'artifacts/parakeet-validated-composition-root-amd-20260924'
GRAPH=ROOT/'artifacts/parakeet-observed-dense-where-graphs-amd-20260924'
PRIOR=dict(build=BUILD,parakeet=ROOT/'artifacts/parakeet-observed-dense-where-app-amd-20260924',
    shared=ROOT/'artifacts/parakeet-observed-dense-where-shared-amd-20260924',
    pyannote_models=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-amd-20260924',
    pyannote=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-app-amd-20260924',
    graphs=GRAPH,selected=SELECTED)
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('product_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def gates():
    verify_scope();source=verify_source()
    for label,folder in PRIOR.items():
        proof=read(folder/'closed.json');assert proof['passed']
        if label in ['parakeet','pyannote','graphs']:assert proof['admitted']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert read(GRAPH/'closed.json')['all_controls_passed']
    assert read(BUILD/'closed.json')['analysis']==pin(BUILD/'analysis.json')
    assert pin(BUILD/'bundle/evidence/source-prepared.json')==pin(SOURCE/'prepared.json')
    assert pin(SELECTED/'closed.json')['sha256']=='c7a1d2e11566e6eeeb965de6c9cedbf47df479fd51f194c797af412446281609'
    product_identities(dict(built=read(BUILD/'analysis.json')['identities']['candidate']),read(SELECTED/'analysis.json'),
        read(PRIOR['parakeet']/'analysis.json'),read(PRIOR['pyannote']/'analysis.json'),read(GRAPH/'payload.json'))
    identities=read(BUILD/'analysis.json')['identities']
    assert read(PRIOR['shared']/'analysis.json')['identities']==identities
    assert read(PRIOR['pyannote_models']/'analysis.json')['identities']==identities
    return source


def previous_closed():
    source=gates();verify_incident();applied=read(APPLIED/'applied.json')
    assert applied['passed'] and applied['source_files']==root_files(source) and applied['prepared']==pin(SOURCE/'prepared.json')
    assert applied['prerequisites']=={label:pin(folder/'closed.json') for label,folder in PRIOR.items()}
    assert applied['graph_qualification']==pin(GRAPH/'closed.json')
    for name,wanted in applied['source_files'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals=verify_scope()
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    applied=read(APPLIED/'applied.json')
    for name in applied['source_files']:copy(ROOT/name,bundle/'source'/name)
    assert sum(name in applied['source_files'] for name in originals)==427
    for p in (SELECTED/'bundle/consumer').iterdir():
        if p.is_file():copy(p,bundle/'consumer'/p.name)
    for name in ['Bridge.dll','Bridge.deps.json','Bridge.runtimeconfig.json']:copy(SELECTED/'bundle/bridge'/name,bundle/'bridge'/name)
    copy(SELECTED/'bundle/evidence/tensor-source.tar',bundle/'evidence/tensor-source.tar')
    for name in ['backend','tensors']:copy(SELECTED/'collected'/(name+'-tests')/(name+'.trx'),bundle/'evidence'/('selected-'+name+'.trx'))
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','new_cases.py','graph_prerequisite.py']:copy(TOOLS/name,bundle/'tools'/name)
    for name in ['closed.json','analysis.json']:copy(GRAPH/name,bundle/'evidence/graph-qualification'/name)
    copy(APPLIED/'applied.json',bundle/'evidence/root-applied.json')
    for name in ['closed.json','failure-analysis.json']:copy(INCIDENT/name,bundle/'evidence/interrupted-root'/name)
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(TOOLS/'README.md',bundle/'prospective-plan.md')
    stage=dict(passed=True,measured=read(BUILD/'analysis.json')['identities']['candidate'],root_integration=pin(APPLIED/'applied.json'),
        graph_qualification=dict(closed=pin(GRAPH/'closed.json'),analysis=pin(GRAPH/'analysis.json')),
        source_scope='427 actual integrated root files: 426 exact measured source inputs plus public mask regression tests.',
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    verify_bundle(bundle,stage);save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),source_files=427)))


if __name__=='__main__':prepare()
