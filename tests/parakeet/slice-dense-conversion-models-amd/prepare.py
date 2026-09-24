"""Bind unchanged full-model checks to the one qualified slice-copy override."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
ORIGINAL=TOOLS.parent/'observed-dense-where-models-amd';sys.path.append(str(ORIGINAL))
from protocol import pin,read,save
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
CURRENT=ROOT/'artifacts/parakeet-observed-dense-where-models-amd-20260924'
PREVIOUS=ROOT/'artifacts/pyannote-winograd-product-parakeet-amd-20260923'
BUILD=ROOT/'artifacts/parakeet-slice-dense-conversion-recovery-amd-20260924'
TESTS=ROOT/'artifacts/parakeet-slice-dense-conversion-tests-amd-20260924'
CONTRACTS=ROOT/'artifacts/parakeet-slice-dense-conversion-isa-amd-20260924'
SOURCE=ROOT/'artifacts/parakeet-slice-dense-conversion-source-20260924'
UNCHANGED=['protocol.py','remote.py','checks.py','native_audit.py','public_audit.py']


def previous_closed():
    for folder,digest in [(CURRENT,'ff9a7a2b5f1156bd365d0f51564f1160e72a82f632ce6e544bb7f86a5663e337'),
        (PREVIOUS,'38b346aec0df29b4e99163282253012a5f066b2d80615cd7517a730b199ac14f'),
        (CONTRACTS,'a4d86d05246f497000639435ba96f583b447da0ecabe0f5510abf4ee3886316f')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed'] and proof['analysis']==pin(folder/'analysis.json')
        for name,wanted in proof.get('files',{}).items():assert pin(folder/name)==wanted,name
    assert pin(SOURCE/'prepared.json')['sha256']=='a9811c6369122eb8b53ab1d88d8329928a999d172d89786a1b393c3451e91e1e'
    source=read(SOURCE/'prepared.json');assert source['passed'] and not source['component_comparison_admitted']
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in source['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    review=read(BUILD/'build-review.json')
    assert pin(BUILD/'build-review.json')['sha256']=='349ae125453183b0b04202dd6408c789760e1748ec863fc51c44d81208f17cbd'
    assert review['passed'] and review['methods']['original']==review['methods']['unchanged']==3253
    assert review['methods']['original_flags_equal'] and review['methods']['effective_conversion_signature_equal']
    assert review['built']==pin(BUILD/'build-collected/built.json') and review['inventory']==pin(BUILD/'build-collected/inventory/instructions.json')
    contracts=read(CONTRACTS/'analysis.json');built=read(BUILD/'build-collected/built.json')
    assert contracts['passed'] and contracts['core']==built['core']==pin(BUILD/'build-collected/runtime/Lokad.Onnx.dll')
    assert contracts['core']['sha256']=='49c3a958850d3e57daa2b7e29e6bd15ff9fc4f27af098d8ce44d8f20b9e065e8'
    assert contracts['compiled_review']==pin(BUILD/'build-review.json') and contracts['corrected_test_review']==pin(TESTS/'build-review.json')
    assert [(s['mode'],s['passed'],s['skipped']) for s in contracts['suites']]==[('512',395,0),('256',395,0)]
    assert not contracts['performance_measured'] and not contracts['product_rebuilt']
    for folder,kind in [(BUILD,'build'),(TESTS,'capture'),(CONTRACTS,'capture')]:
        collected=folder/(kind+'-collected');receipt=read(collected/(kind+'-collection.json'))
        assert receipt['terminal']
        for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    expected=(ORIGINAL/'audit.py').read_text().replace('M70 current release','M73 current release').replace('M70 observed-mask composition','M73 slice dense conversion')
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
    for name in ['closed.json','analysis.json']:copy(CONTRACTS/name,bundle/'evidence'/('contracts-'+name))
    copy(BUILD/'build-review.json',bundle/'evidence/build-review.json')
    copy(TESTS/'build-review.json',bundle/'evidence/test-review.json')
    terminals=[]
    for folder,kind,label in [(BUILD,'build','build'),(TESTS,'capture','ordinary'),(CONTRACTS,'capture','disabled')]:
        remote='/dev/shm/lokad-'+folder.name.removesuffix('-amd-20260924')+'-20260924'
        for name in [kind+'-collection.json',kind+'-state.json']:
            copy(folder/(kind+'-collected')/name,bundle/'evidence'/(label+'-'+name))
        terminals.append(dict(remote=remote,kind=kind,label=label))
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(CURRENT/'collected/manifests/candidate-parakeet.json',bundle/'evidence/original-manifest.json')
    identities={}
    for role in ['selected','candidate']:
        identities[role]={}
        for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']:
            path=(BUILD/'build-collected/runtime'/name if role=='candidate' and name=='Lokad.Onnx.dll'
                else CURRENT/'collected/runtimes/candidate'/name)
            identities[role][name]=pin(path);copy(path,bundle/'products'/role/name)
    assert identities['selected']==read(CURRENT/'analysis.json')['identities']['candidate']
    assert identities['candidate']['Lokad.Onnx.dll']==read(CONTRACTS/'analysis.json')['core']
    assert identities['candidate']['Lokad.Onnx.Data.dll']==identities['selected']['Lokad.Onnx.Data.dll']
    consumers={name:pin(CURRENT/'collected/runtimes/candidate'/(name+'.dll')) for name in ['TranscribeReplay','AudioBenchmark']}
    assert consumers==read(CURRENT/'analysis.json')['consumers']
    copy(TOOLS/'README.md',bundle/'prospective-models.md')
    copy(ROOT/'.agent/m73-parakeet-slice-dense-conversion-20260924.md',bundle/'prospective-plan.md')
    save(bundle/'stage.json',dict(passed=True,identities=identities,consumers=consumers,terminal_sources=terminals,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    for name in ['run.py','audit.py',*UNCHANGED]:originals[(ORIGINAL/name).relative_to(ROOT).as_posix()]=pin(ORIGINAL/name)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    originals.pop('.agent/m73-parakeet-slice-dense-conversion-20260924.md')
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),identities=identities)))


if __name__=='__main__':prepare()
