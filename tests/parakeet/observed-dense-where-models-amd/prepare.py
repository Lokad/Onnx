"""Bind unchanged full-model qualification to the observed-mask composition."""
import ast
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-observed-dense-where-models-amd-20260924'
BUILD=ROOT/'artifacts/parakeet-observed-dense-where-inventory-amd-20260924'
CONTRACTS=ROOT/'artifacts/parakeet-observed-dense-where-numerics-amd-20260924'
CURRENT=ROOT/'artifacts/parakeet-validated-composition-models-amd-20260924'
PREVIOUS=ROOT/'artifacts/pyannote-winograd-product-parakeet-amd-20260923'
SOURCE=ROOT/'artifacts/parakeet-observed-dense-where-source-20260924'
ORIGINAL=TOOLS.parent/'validated-composition-models-amd'


def previous_closed():
    for folder,digest in [(BUILD,'7d5296b266feeacbf9c52bf57a689de2cdf0a839045130f15912912f5da1b7bd'),
        (CONTRACTS,'0dfb5dcb4d5c5793c7c175837bde6f9469e791f62c02cf1b4df679ba4d81f0db'),
        (CURRENT,'a71906951cccb561c71b747a12f815d8145efb1d7bd7dabcb61689bdf9bb3802'),
        (PREVIOUS,'38b346aec0df29b4e99163282253012a5f066b2d80615cd7517a730b199ac14f')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        if 'analysis' in proof:assert pin(folder/'analysis.json')==proof['analysis']
        for name,wanted in proof.get('files',{}).items():assert pin(folder/name)==wanted,name
    assert pin(SOURCE/'prepared.json')['sha256']=='41f2477c7060c3f543471bebe991c77ea3b853d724708fec098a2cbb505cb0a2'
    source=read(SOURCE/'prepared.json');assert source['passed']
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in source['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    build=read(BUILD/'analysis.json');numerics=read(CONTRACTS/'analysis.json')
    assert build['source_prepared']==pin(SOURCE/'prepared.json')
    assert build['measured']==read(CURRENT/'analysis.json')['identities']['candidate']==source['measured']
    assert numerics['products']==dict(current=build['measured'],candidate=build['built'])
    assert numerics['cases_per_boundary']==235 and numerics['boundaries']==2 and numerics['numerical_workers']==4
    assert numerics['values_all_numerical_workers']==30403936 and numerics['all_selected_bits_exact']
    assert numerics['input_ownership_and_held_outputs'] and numerics['original_131_case_hashes_exact']
    assert build['inventory']['retained_provider_and_helper_bodies_exact'] and build['inventory']['unchanged_core_methods']==3250
    for name in ['protocol.py','remote.py','checks.py','native_audit.py','public_audit.py']:
        assert (TOOLS/name).read_bytes()==(ORIGINAL/name).read_bytes(),name
    expected=(ORIGINAL/'audit.py').read_text().replace('M66 selected release','M70 current release').replace('M66 admitted-source composition','M70 observed-mask composition')
    assert (TOOLS/'audit.py').read_text()==expected


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','native_audit.py','public_audit.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for folder,label in [(CURRENT,'current'),(PREVIOUS,'previous'),(BUILD,'build'),(CONTRACTS,'contracts')]:
        for name in ['closed.json','payload.json']:copy(folder/name,bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
    for folder,label in [(BUILD,'build'),(CONTRACTS,'contracts')]:
        copy(folder/'analysis.json',bundle/'evidence'/(label+'-analysis.json'))
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(CURRENT/'collected/manifests/candidate-parakeet.json',bundle/'evidence/original-manifest.json')
    identities={}
    for role in ['selected','candidate']:
        identities[role]={}
        for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']:
            path=(CURRENT/'collected/runtimes/candidate' if role=='selected' else BUILD/'collected/runtime')/name
            identities[role][name]=pin(path);copy(path,bundle/'products'/role/name)
    consumers={name:pin(CURRENT/'collected/runtimes/candidate'/(name+'.dll')) for name in ['TranscribeReplay','AudioBenchmark']}
    assert consumers['TranscribeReplay']['sha256']=='335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
    assert consumers['AudioBenchmark']['sha256']=='7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
    numerics=read(CONTRACTS/'analysis.json')['products']
    assert identities==dict(selected=numerics['current'],candidate=numerics['candidate'])
    copy(TOOLS/'README.md',bundle/'prospective-models.md')
    copy(ROOT/'.agent/m70-parakeet-observed-dense-where-20260924.md',bundle/'prospective-plan.md')
    save(bundle/'stage.json',dict(passed=True,identities=identities,consumers=consumers,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    for name in ['protocol.py','remote.py','checks.py','native_audit.py','public_audit.py','audit.py']:
        originals[(ORIGINAL/name).relative_to(ROOT).as_posix()]=pin(ORIGINAL/name)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    originals.pop('.agent/m70-parakeet-observed-dense-where-20260924.md')
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(__import__('json').dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),identities=identities)))


if __name__=='__main__':prepare()
