"""Bind the unchanged full-model protocol to the qualified direct-depthwise candidate."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
ORIGINAL=TOOLS.parent/'observed-dense-where-models-amd';sys.path.append(str(ORIGINAL))
from protocol import pin,read,save
BASE=ROOT/'artifacts/parakeet-direct-depthwise-models-amd-20260925'
CURRENT=ROOT/'artifacts/parakeet-packed-final-row-models-amd-20260925'
PREVIOUS=ROOT/'artifacts/pyannote-winograd-product-parakeet-amd-20260923'
CONTRACTS=ROOT/'artifacts/parakeet-direct-depthwise-build-v2-amd-20260925'
CENSUS=ROOT/'artifacts/parakeet-direct-depthwise-observer-amd-20260925'
UNCHANGED=['protocol.py','remote.py','checks.py','native_audit.py','public_audit.py']
SELECTED_LABEL='Direct depthwise selected M78'
CANDIDATE_LABEL='Direct nine-tap depthwise'


def previous_closed():
    for folder,digest in [(CURRENT,'1a0da5fc'),(PREVIOUS,'38b346ae'),(CONTRACTS,'26e4da0a'),(CENSUS,'ac821813')]:
        assert pin(folder/'closed.json')['sha256'].startswith(digest)
        proof=read(folder/'closed.json');assert proof['passed'] and proof['analysis']==pin(folder/'analysis.json')
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    tests=read(CONTRACTS/'analysis.json');census=read(CENSUS/'analysis.json');compiled=read(CONTRACTS/'build-review.json')
    binding=read(CENSUS/'build-review.json');observed_spec=read(CENSUS/'bundle/spec.json')
    assert compiled['passed'] and tests['compiled_review']==pin(CONTRACTS/'build-review.json')
    assert compiled['product']==tests['product']==observed_spec['before_product']
    observer_source=ROOT/'artifacts/parakeet-direct-depthwise-observer-source-20260925/prepared.json'
    assert pin(observer_source)==observed_spec['source']==census['source']
    assert read(observer_source)['baseline']==compiled['source']
    assert [len(r['changed']) for r in compiled['methods']]==[1,0]
    assert len(compiled['methods'][0]['added'])==4 and compiled['data_binary_unchanged'] and compiled['zero_added_warnings']
    assert [(s['mode'],s['passed'],s['skipped']) for s in tests['suites']]==[('normal',8,0),('scalar',8,0)]
    assert census['diagnostic_only'] and census['exact_public_results'] and census['public_requests']==80
    assert census['observed']['every_geometry_exact'] and census['observed']['zero_generic_work']
    assert census['observed']['per_corpus']['direct_batches']==520
    assert binding['passed'] and binding['products']==census['products']
    expected=(ORIGINAL/'audit.py').read_text().replace('M70 current release',SELECTED_LABEL).replace('M70 observed-mask composition',CANDIDATE_LABEL)
    assert (TOOLS/'audit.py').read_text()==expected


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in UNCHANGED:copy(ORIGINAL/name,bundle/'tools'/name)
    copy(TOOLS/'remote_prepare.py',bundle/'tools/remote_prepare.py')
    for folder,label in [(CURRENT,'current'),(PREVIOUS,'previous')]:
        for name in ['closed.json','payload.json']:copy(folder/name,bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
    for folder,label in [(CONTRACTS,'contracts'),(CENSUS,'census')]:
        for name in ['closed.json','analysis.json']:copy(folder/name,bundle/'evidence'/(label+'-'+name))
    copy(CENSUS/'bundle/spec.json',bundle/'evidence/observer-spec.json')
    copy(CONTRACTS/'build-review.json',bundle/'evidence/build-review.json')
    copy(CENSUS/'build-review.json',bundle/'evidence/census-build-review.json')
    terminals=[]
    for folder,kind,label in [(CONTRACTS,'build','build'),(CONTRACTS,'capture','contracts'),(CENSUS,'capture','census')]:
        remote='/dev/shm/lokad-'+folder.name.removesuffix('-amd-20260925')+'-20260925'
        for name in [kind+'-collection.json',kind+'-state.json']:
            copy(folder/(kind+'-collected')/name,bundle/'evidence'/(label+'-'+name))
        terminals.append(dict(remote=remote,kind=kind,label=label))
    copy(CURRENT/'collected/manifests/candidate-parakeet.json',bundle/'evidence/original-manifest.json')
    identities={}
    for role in ['selected','candidate']:
        identities[role]={}
        for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']:
            path=(CONTRACTS/'build-collected/runtime'/name if role=='candidate' else CURRENT/'collected/runtimes/candidate'/name)
            identities[role][name]=pin(path);copy(path,bundle/'products'/role/name)
    assert identities['selected']==read(CURRENT/'analysis.json')['identities']['candidate']
    assert identities['candidate']==read(CONTRACTS/'analysis.json')['product']
    assert identities['selected']['Lokad.Onnx.dll']['sha256']=='49901366484570493b7a42028e5c5458f30fe2c1703a01335b68d9a5f9a1fea9'
    consumers={name:pin(CURRENT/'collected/runtimes/candidate'/(name+'.dll')) for name in ['TranscribeReplay','AudioBenchmark']}
    assert consumers==read(CURRENT/'analysis.json')['consumers']
    copy(TOOLS/'README.md',bundle/'prospective-models.md')
    copy(ROOT/'PLAN.md',bundle/'prospective-plan.md')
    save(bundle/'stage.json',dict(passed=True,identities=identities,consumers=consumers,terminal_sources=terminals,
        failed_release_controls=['M78 e5-8tok release regression 1.0776541557'],release_admitted=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    for name in ['run.py','audit.py',*UNCHANGED]:originals[(ORIGINAL/name).relative_to(ROOT).as_posix()]=pin(ORIGINAL/name)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    originals.pop('PLAN.md')
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),identities=identities)))


if __name__=='__main__':prepare()
