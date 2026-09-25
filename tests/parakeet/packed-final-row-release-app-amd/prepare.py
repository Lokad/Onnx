"""Bind the unchanged application protocol after actual owned-weight accounting."""
import ast
import json
import shutil
import tarfile
from pathlib import Path
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-packed-final-row-release-app-amd-20260925'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
MODELS=ROOT/'artifacts/parakeet-packed-final-row-models-amd-20260925'
CONTRACTS=ROOT/'artifacts/parakeet-packed-final-row-build-amd-20260925'
CENSUS=ROOT/'artifacts/parakeet-packed-final-row-census-resume-amd-20260925'
COUNTERS=ROOT/'artifacts/parakeet-packed-final-row-counters-amd-20260925'
SELECTED_APP=ROOT/'artifacts/parakeet-slice-dense-conversion-app-amd-20260925'
RELEASE_MODELS=ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
CANDIDATE_APP=ROOT/'artifacts/parakeet-packed-final-row-app-amd-20260925'
PRIOR=dict(baseline=CURRENT,models=MODELS,contracts=CONTRACTS,census=CENSUS,counters=COUNTERS,
           selected_app=SELECTED_APP,release_models=RELEASE_MODELS,candidate_app=CANDIDATE_APP)
DIGESTS=dict(baseline='2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3',
    models='1a0da5fcd612d893c4954c6c2761a358af081f2a693e462ee138743fad29777d',counters='a50f8afaff3d3742f4974e1d306caaf11350f14c542a062a87f21cfaa2371ce4',
    contracts='6295ad30835b7a2a1694b580a8e1447a6a0cdb28828a5960fa6e0e7c3628576f',
    census='fa159300cbe217b8ac08df793031850f2794f2149dc65b89e89b93a5d5692a77',
    selected_app='c9e72e572118d4ea8e23c3438e53e77a6529f9f4c35c5f93919c27c9d7932326',
    release_models='5860fc366c3d976907f35f8d7a1832c10566fa08910af32c168dca7a63979842',
    candidate_app='b90aa8fc3949c756cc7c55e636d2a8741c31fdb8d336174502639a323ebc0c06')


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
        assert row['avoided_packs']==1740 and row['reconstructions']==0 and row['scratch_reduction']==29192355840 and row['copy_increase']==0
        assert len(row['clips'])==20 and all(c['outputs_exact'] for c in row['clips'])
    assert not counters['application_scored'] and not counters['release_admitted']
    assert read(CONTRACTS/'analysis.json')['compiled_review']==pin(CONTRACTS/'build-review.json')
    assert read(CENSUS/'analysis.json')['contracts']==pin(CONTRACTS/'closed.json')
    assert read(CENSUS/'analysis.json')['original_compiled_review']==pin(CONTRACTS/'build-review.json')
    assert read(CANDIDATE_APP/'closed.json')['admitted']
    assert read(CANDIDATE_APP/'analysis.json')['performance']['admitted']


def prepare():
    assert not BASE.exists();previous_closed();assert (ROOT/'PLAN.md').is_file()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    originals={};prerequisites={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','lineage.py','statistics_exact.py']:copy(TOOLS/name,bundle/'tools'/name)
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json']:copy(folder/name,bundle/'evidence'/label/name)
        prerequisites[label]=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    for label,folder in [('baseline',CURRENT),('models',MODELS),('release_models',RELEASE_MODELS)]:
        copy(folder/'payload.json',bundle/'evidence'/label/'payload.json')
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    for source,name in [('bundle/spec.json','spec.json'),('capture-collected/capture-collection.json','collection.json'),
                        ('capture-collected/capture-state.json','state.json'),('build-review.json','build-review.json')]:
        copy(COUNTERS/source,bundle/'evidence/counters'/name)
    copy(CONTRACTS/'build-review.json',bundle/'evidence/contracts/build-review.json')
    for role,folder,original in [('current',RELEASE_MODELS,'selected'),('candidate',MODELS,'candidate')]:
        copy(folder/'collected'/(original+'-public-512/output/result.json'),bundle/'evidence'/(role+'-public.json'))
        copy(folder/'collected/manifests'/(original+'-parakeet.json'),bundle/'evidence'/(role+'-parakeet.json'))
    for generation,folder in [('release',RELEASE_MODELS),('m78',MODELS)]:
        for role in ['selected','candidate']:
            for isa in ['512','256']:
                copy(folder/'collected'/(role+'-public-'+isa)/'output/result.json',
                     bundle/'evidence/public-lineage'/generation/(role+'-'+isa+'.json'))
    copy(TOOLS/'README.md',bundle/'prospective-application.md')
    shutil.copy2(ROOT/'PLAN.md',bundle/'prospective-plan.md')
    models=read(MODELS/'analysis.json')
    stage=dict(passed=True,identities=dict(current=read(RELEASE_MODELS/'analysis.json')['identities']['selected'],candidate=models['identities']['candidate']),
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
    for name in ['checks.py','audit.py']:
        path=TOOLS.parent/'packed-final-row-app-amd'/name;originals[path.relative_to(ROOT).as_posix()]=pin(path)
    for name in ['lineage.py','test_lineage.py']:
        path=TOOLS.parent/'packed-final-row-pyannote-app-amd'/name;originals[path.relative_to(ROOT).as_posix()]=pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
