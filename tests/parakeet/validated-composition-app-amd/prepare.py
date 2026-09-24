"""Freeze the original application comparison after matched pair attribution."""
import ast
import json
import shutil
import tarfile
from pathlib import Path
from protocol import pin, read, save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-validated-composition-app-amd-20260924'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS=ROOT/'artifacts/parakeet-validated-composition-models-amd-20260924'
COMPOSITION=ROOT/'artifacts/parakeet-validated-composition-build-amd-v2-20260924'
QUALIFIED=ROOT/'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
SOURCE=ROOT/'artifacts/parakeet-validated-composition-source-v2-20260924'
RECURRENCE=ROOT/'artifacts/parakeet-prepared-recurrence-app-amd-20260924'
SLICE=ROOT/'artifacts/parakeet-slice-materialization-app-amd-20260924'
PRIOR=dict(baseline=CURRENT,models=MODELS,composition=COMPOSITION,root=QUALIFIED,
    recurrence=RECURRENCE,slice=SLICE)
DIGESTS=dict(baseline='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3',
    recurrence='36b7809016735de7bdeee3b9c5c15c984ca0364c560e3069cc369e05de56ecf0',
    root='16d570819ab69915fe34fa6c5a4efb79d45ae792dbd5b1448c65645fa0d55f73')


def previous_closed():
    from consumer_scope import verify_scope
    assert verify_scope()
    for name,folder in PRIOR.items():
        proof=read(folder/'closed.json');assert proof['passed']
        if name in DIGESTS:assert pin(folder/'closed.json')['sha256']==DIGESTS[name]
        assert proof['analysis']==pin(folder/'analysis.json')
        root=ROOT if proof.get('paths_relative_to_repository') else folder
        for path,wanted in proof.get('files',{}).items():assert pin(root/path)==wanted,path
    source=read(SOURCE/'prepared.json');assert source['passed'] and source['all_product_parent_bytes_preserved']
    assert read(COMPOSITION/'analysis.json')['source_prepared']==pin(SOURCE/'prepared.json')
    for label,folder in [('recurrence',RECURRENCE),('slice',SLICE)]:
        assert read(folder/'closed.json')['admitted']
        assert source['admissions'][label]['closure']==pin(folder/'closed.json')
        assert source['admissions'][label]['identities']==read(folder/'analysis.json')['identities']
    assert read(COMPOSITION/'closed.json')['build_review']==pin(COMPOSITION/'build-review.json')
    assert read(COMPOSITION/'build-review.json')['built']==pin(COMPOSITION/'build-collected/built.json')
    for path,wanted in source['source'].items():assert pin(SOURCE/'source'/path)==wanted,path
    for path,wanted in source['before'].items():assert pin(ROOT/path)==wanted,path


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    originals={};prerequisites={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','statistics_exact.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json']:copy(folder/name,bundle/'evidence'/label/name)
        prerequisites[label]=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    for label,folder in [('baseline',CURRENT),('models',MODELS)]:
        copy(folder/'payload.json',bundle/'evidence'/label/'payload.json')
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    for source,name in [('bundle/spec.json','spec.json'),('build-collected/built.json','built.json'),
        ('build-review.json','build-review.json'),('capture-collected/capture-collection.json','collection.json'),
        ('capture-collected/capture-state.json','state.json')]:
        copy(COMPOSITION/source,bundle/'evidence/composition'/name)
    copy(SOURCE/'prepared.json',bundle/'evidence/composition/source-prepared.json')
    for role,original in [('current','selected'),('candidate','candidate')]:
        copy(MODELS/'collected'/(original+'-public-512/output/result.json'),bundle/'evidence'/(role+'-public.json'))
        copy(MODELS/'collected/manifests'/(original+'-parakeet.json'),bundle/'evidence'/(role+'-parakeet.json'))
    copy(TOOLS/'README.md',bundle/'prospective-application.md')
    shutil.copy2(ROOT/'.agent/m66-parakeet-validated-composition-20260924.md',bundle/'prospective-plan.md')
    models=read(MODELS/'analysis.json')
    stage=dict(passed=True,identities=dict(current=models['identities']['selected'],candidate=models['identities']['candidate']),
        prerequisites=prerequisites,consumers=dict(AudioBenchmark=models['consumers']['AudioBenchmark']),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
