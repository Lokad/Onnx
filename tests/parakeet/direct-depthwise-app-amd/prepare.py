"""Prepare one full-transcription decision for the numerically qualified candidate."""
import ast
import json
import shutil
import tarfile
from pathlib import Path
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-direct-depthwise-app-amd-20260925'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS=ROOT/'artifacts/parakeet-direct-depthwise-models-amd-20260925'
CONTRACTS=ROOT/'artifacts/parakeet-direct-depthwise-build-v2-amd-20260925'
MECHANISM=ROOT/'artifacts/parakeet-direct-depthwise-observer-amd-20260925'
SELECTED=ROOT/'artifacts/parakeet-packed-final-row-release-app-amd-20260925'
GRAPHS=ROOT/'artifacts/parakeet-packed-final-row-graphs-amd-20260925'
PRIOR=dict(baseline=CURRENT,models=MODELS,contracts=CONTRACTS,mechanism=MECHANISM,selected_app=SELECTED,graphs=GRAPHS)
DIGESTS=dict(baseline='2e75c249',models='1dd72087',contracts='26e4da0a',mechanism='ac821813',selected_app='db3dd71f',graphs='0b82805a')


def previous_closed():
    from consumer_scope import verify_scope
    assert verify_scope()
    for name,folder in PRIOR.items():
        proof=read(folder/'closed.json');assert proof['passed']
        assert pin(folder/'closed.json')['sha256'].startswith(DIGESTS[name])
        recorded=proof['files']['analysis.json'] if name=='graphs' else proof['analysis']
        assert recorded==pin(folder/'analysis.json')
        root=ROOT if proof.get('paths_relative_to_repository') else folder
        for path,wanted in proof.get('files',{}).items():assert pin(root/path)==wanted,path
    assert read(SELECTED/'closed.json')['admitted']
    assert not read(GRAPHS/'closed.json')['admitted']


def prepare():
    assert not BASE.exists();previous_closed()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={};prerequisites={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','prerequisites.py','statistics_exact.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json']:copy(folder/name,bundle/'evidence'/label/name)
        prerequisites[label]=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    for label,folder in [('baseline',CURRENT),('models',MODELS)]:
        copy(folder/'payload.json',bundle/'evidence'/label/'payload.json')
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    copy(CONTRACTS/'build-review.json',bundle/'evidence/contracts/build-review.json')
    copy(MECHANISM/'bundle/spec.json',bundle/'evidence/mechanism/spec.json')
    copy(ROOT/'artifacts/parakeet-direct-depthwise-observer-source-20260925/prepared.json',
         bundle/'evidence/mechanism/source-prepared.json')
    for role,source in [('current','selected'),('candidate','candidate')]:
        copy(MODELS/'collected'/(source+'-public-512/output/result.json'),bundle/'evidence'/(role+'-public.json'))
        copy(MODELS/'collected/manifests'/(source+'-parakeet.json'),bundle/'evidence'/(role+'-parakeet.json'))
    copy(TOOLS/'README.md',bundle/'prospective-application.md')
    shutil.copy2(ROOT/'PLAN.md',bundle/'prospective-plan.md')
    models=read(MODELS/'analysis.json')
    stage=dict(passed=True,identities=dict(current=models['identities']['selected'],candidate=models['identities']['candidate']),
        prerequisites=prerequisites,consumers=dict(AudioBenchmark=models['consumers']['AudioBenchmark']),
        failed_graph_cases=read(CONTRACTS/'analysis.json')['failed_graph_cases'],release_admitted=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    from checks import prereqs
    prereqs(bundle,stage);save(bundle/'stage.json',stage)
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    for folder,names in [('observed-dense-where-app-amd',['checks.py','protocol.py','remote.py','statistics_exact.py','test_admission.py','audit.py']),
                         ('packed-final-row-release-app-amd',['run.py'])]:
        for name in names:
            path=TOOLS.parent/folder/name;originals[path.relative_to(ROOT).as_posix()]=pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
