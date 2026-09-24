"""Freeze existing model consumers with the reviewed slice-copy products."""
import ast
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-validated-composition-models-amd-20260924'
LAYOUT=ROOT/'artifacts/parakeet-slice-layout-amd-20260924'
CONTRACTS=ROOT/'artifacts/parakeet-validated-composition-build-amd-v2-20260924'
BUILD=ROOT/'artifacts/parakeet-validated-composition-build-amd-v2-20260924'
CURRENT=ROOT/'artifacts/parakeet-wide-entry-first-use-models-amd-20260923'
PREVIOUS=ROOT/'artifacts/pyannote-winograd-product-parakeet-amd-20260923'
SOURCE=ROOT/'artifacts/parakeet-validated-composition-source-v2-20260924'


def previous_closed():
    for folder,digest in [(LAYOUT,'664cde30a533dd56ff13eeb149289a8e1fc42ae89a0689bd44e46dbec039ca95'),
        (CURRENT,'f30100534cbb79db790aac30365d533e6d3dcce79e779abc5feb8f7cf3fc1e22'),
        (PREVIOUS,'38b346aec0df29b4e99163282253012a5f066b2d80615cd7517a730b199ac14f')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        if 'analysis' in proof:assert pin(folder/'analysis.json')==proof['analysis']
        for name,wanted in proof.get('files',{}).items():assert pin(folder/name)==wanted,name
    proof=read(BUILD/'closed.json');assert proof['passed'] and proof['analysis']==pin(BUILD/'analysis.json')
    assert proof['build_review']==pin(BUILD/'build-review.json')
    review=read(BUILD/'build-review.json');assert review['passed']
    assert review['built']==pin(BUILD/'build-collected/built.json')
    built=read(BUILD/'build-collected/built.json')
    for name,wanted in built['runtime_files'].items():assert pin(BUILD/'build-collected'/name)==wanted,name
    qualification=read(CONTRACTS/'analysis.json')
    assert qualification['core']==built['core'] and qualification['same_binaries']
    assert all(s['passed']==369 and s['skipped']==0 for s in qualification['suites'])
    assert review['source_prepared']==pin(SOURCE/'prepared.json')
    assert review['methods']['qualified_copy_bodies_exact'] and review['methods']['unchanged']==3249
    source=read(SOURCE/'prepared.json')
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in source['source'].items():assert pin(SOURCE/'source'/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','native_audit.py','public_audit.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for name,original in [('native_audit.py',ROOT/'tests/parakeet/transcribe/audit.py'),
        ('public_audit.py',ROOT/'tests/audio/comparison/audit.py'),
        ('checks.py',ROOT/'tests/parakeet/prepared-recurrence-models-amd/checks.py'),
        ('remote.py',ROOT/'tests/parakeet/prepared-recurrence-models-amd/remote.py')]:
        assert (TOOLS/name).read_bytes()==original.read_bytes();originals[original.relative_to(ROOT).as_posix()]=pin(original)
    for folder,label in [(CURRENT,'current'),(PREVIOUS,'previous')]:
        for name in ['closed.json','payload.json']:copy(folder/name,bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
    for folder,label in [(LAYOUT,'layout'),(CONTRACTS,'contracts')]:
        for name in ['closed.json','analysis.json']:copy(folder/name,bundle/'evidence'/(label+'-'+name))
    copy(LAYOUT/'capture-collected/capture-collection.json',bundle/'evidence/layout-collection.json')
    copy(CONTRACTS/'capture-collected/capture-collection.json',bundle/'evidence/contracts-collection.json')
    copy(BUILD/'build-review.json',bundle/'evidence/build-review.json')
    copy(BUILD/'build-collected/built.json',bundle/'evidence/built.json')
    copy(BUILD/'build-collected/build-collection.json',bundle/'evidence/build-collection.json')
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(CURRENT/'collected/manifests/candidate-parakeet.json',bundle/'evidence/original-manifest.json')
    identities={}
    for role in ['selected','candidate']:
        identities[role]={}
        for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']:
            path=CURRENT/'collected/runtimes/candidate'/name
            if role=='candidate':path=BUILD/'build-collected/source/runtime-observed'/name
            identities[role][name]=pin(path);copy(path,bundle/'products'/role/name)
    consumers={name:pin(CURRENT/'collected/runtimes/candidate'/(name+'.dll')) for name in ['TranscribeReplay','AudioBenchmark']}
    assert consumers['TranscribeReplay']['sha256']=='335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
    assert consumers['AudioBenchmark']['sha256']=='7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
    assert identities['selected']==read(CURRENT/'analysis.json')['identities']['candidate']
    assert identities['candidate']['Lokad.Onnx.dll']==read(CONTRACTS/'analysis.json')['core']
    assert identities['candidate']['Lokad.Onnx.Data.dll']==read(CONTRACTS/'analysis.json')['data']
    copy(TOOLS/'README.md',bundle/'prospective-models.md')
    copy(ROOT/'.agent/m66-parakeet-validated-composition-20260924.md',bundle/'prospective-plan.md')
    save(bundle/'stage.json',dict(passed=True,identities=identities,consumers=consumers,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    # The plan is a frozen snapshot, not a live prerequisite: progress updates
    # must not invalidate an in-flight immutable campaign.
    originals.pop('.agent/m66-parakeet-validated-composition-20260924.md')
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(__import__('json').dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),identities=identities)))


if __name__=='__main__':prepare()
