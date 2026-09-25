"""Bind the unchanged application protocol after actual owned-weight accounting."""
import ast
import json
import shutil
import tarfile
from pathlib import Path
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-app-amd-20260925'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS=ROOT/'artifacts/parakeet-owned-packed-weight-models-amd-20260925'
CONTRACTS=ROOT/'artifacts/parakeet-owned-packed-weight-scope-recovery-amd-20260925'
CENSUS=ROOT/'artifacts/parakeet-owned-packed-weight-scope-census-amd-20260925'
COUNTERS=ROOT/'artifacts/parakeet-owned-packed-weight-counters-resume-amd-20260925'
SELECTED_APP=ROOT/'artifacts/parakeet-slice-dense-conversion-app-amd-20260925'
PRIOR=dict(baseline=CURRENT,models=MODELS,contracts=CONTRACTS,census=CENSUS,counters=COUNTERS,selected_app=SELECTED_APP)
DIGESTS=dict(baseline='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3',
    models='bb1a758c6936fa251d921992c67dd7c65ff075fe20829af72f4455b529f1b2ca',
    contracts='0f8bdc94e2aaf27956af6767093acdb8b57c31655713c9af1a6302d2f7aa31ae',
    census='577a07a60d901cf57c4ba46d70b2c6fc4c02148ab67a1c28fd78d9b4fbcc3d08',
    selected_app='c9e72e572118d4ea8e23c3438e53e77a6529f9f4c35c5f93919c27c9d7932326')


def previous_closed():
    from consumer_scope import verify_scope
    assert verify_scope()
    for name,folder in PRIOR.items():
        proof=read(folder/'closed.json');assert proof['passed']
        if name in DIGESTS:assert pin(folder/'closed.json')['sha256']==DIGESTS[name]
        assert proof['analysis']==pin(folder/'analysis.json')
        root=ROOT if proof.get('paths_relative_to_repository') else folder
        for path,wanted in proof.get('files',{}).items():assert pin(root/path)==wanted,path
    counters=read(COUNTERS/'analysis.json')
    assert counters['model_closure']==pin(MODELS/'closed.json') and counters['census_closure']==pin(CENSUS/'closed.json')
    assert counters['build_review']==pin(COUNTERS/'build-review.json')
    assert read(COUNTERS/'build-review.json')['passed'] and not read(COUNTERS/'build-review.json')['product_rebuilt']
    assert len(counters['comparisons'])==2
    for row in counters['comparisons']:
        assert row['avoided_packs']==1740 and row['reconstructions']==609 and row['scratch_reduction']==29192355840 and row['copy_increase']==10217324544
        assert len(row['clips'])==20 and all(c['outputs_exact'] for c in row['clips'])
    assert not counters['application_scored'] and not counters['release_admitted']
    assert read(CONTRACTS/'analysis.json')['compiled_review']==pin(CONTRACTS/'build-review.json')
    assert read(CENSUS/'analysis.json')['contracts']==pin(CONTRACTS/'closed.json')
    assert read(CENSUS/'analysis.json')['original_compiled_review']==pin(CONTRACTS/'build-review.json')


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    originals={};prerequisites={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','statistics_exact.py']:copy(TOOLS/name,bundle/'tools'/name)
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json']:copy(folder/name,bundle/'evidence'/label/name)
        prerequisites[label]=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    for label,folder in [('baseline',CURRENT),('models',MODELS)]:
        copy(folder/'payload.json',bundle/'evidence'/label/'payload.json')
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    for source,name in [('bundle/spec.json','spec.json'),('capture-collected/capture-collection.json','collection.json'),
                        ('capture-collected/capture-state.json','state.json'),('build-review.json','build-review.json')]:
        copy(COUNTERS/source,bundle/'evidence/counters'/name)
    copy(CONTRACTS/'build-review.json',bundle/'evidence/contracts/build-review.json')
    for role,original in [('current','selected'),('candidate','candidate')]:
        copy(MODELS/'collected'/(original+'-public-512/output/result.json'),bundle/'evidence'/(role+'-public.json'))
        copy(MODELS/'collected/manifests'/(original+'-parakeet.json'),bundle/'evidence'/(role+'-parakeet.json'))
    copy(TOOLS/'README.md',bundle/'prospective-application.md')
    shutil.copy2(ROOT/'.agent/m76-parakeet-owned-packed-weights-20260925.md',bundle/'prospective-plan.md')
    models=read(MODELS/'analysis.json')
    stage=dict(passed=True,identities=dict(current=models['identities']['selected'],candidate=models['identities']['candidate']),
        prerequisites=prerequisites,consumers=dict(AudioBenchmark=models['consumers']['AudioBenchmark']),
        failed_release_controls=read(COUNTERS/'analysis.json')['failed_release_controls'],release_admitted=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    from checks import prereqs
    prereqs(bundle,stage);save(bundle/'stage.json',stage)
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    prior=TOOLS.parent/'observed-dense-where-app-amd'
    for name in ['checks.py','remote.py','statistics_exact.py','test_admission.py','audit.py','protocol.py']:originals[(prior/name).relative_to(ROOT).as_posix()]=pin(prior/name)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
