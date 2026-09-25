"""Bind the unchanged full-model protocol to the single owned-weight candidate."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
ORIGINAL=TOOLS.parent/'observed-dense-where-models-amd';sys.path.append(str(ORIGINAL))
from protocol import pin,read,save
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-models-amd-20260925'
CURRENT=ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
PREVIOUS=ROOT/'artifacts/pyannote-winograd-product-parakeet-amd-20260923'
CONTRACTS=ROOT/'artifacts/parakeet-owned-packed-weight-scope-recovery-amd-20260925'
CENSUS=ROOT/'artifacts/parakeet-owned-packed-weight-scope-census-amd-20260925'
UNCHANGED=['protocol.py','remote.py','checks.py','native_audit.py','public_audit.py']
SELECTED_LABEL='M76 selected isolated M73'
CANDIDATE_LABEL='M76 owned packed weights'


def previous_closed():
    for folder,digest in [(CURRENT,'5860fc366c3d976907f35f8d7a1832c10566fa08910af32c168dca7a63979842'),
        (PREVIOUS,'38b346aec0df29b4e99163282253012a5f066b2d80615cd7517a730b199ac14f'),
        (CONTRACTS,'0f8bdc94e2aaf27956af6767093acdb8b57c31655713c9af1a6302d2f7aa31ae'),
        (CENSUS,'577a07a60d901cf57c4ba46d70b2c6fc4c02148ab67a1c28fd78d9b4fbcc3d08')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed'] and proof['analysis']==pin(folder/'analysis.json')
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    tests=read(CONTRACTS/'analysis.json');census=read(CENSUS/'analysis.json');compiled=read(CONTRACTS/'build-review.json')
    assert pin(CONTRACTS/'build-review.json')['sha256']=='5a84c53bd547378f39a0e47ccd7270dd40c346c7eebb5ff7fb4f81ff8fb1689b'
    assert compiled['passed'] and tests['compiled_review']==census['original_compiled_review']==pin(CONTRACTS/'build-review.json')
    assert compiled['product']==tests['product']==census['product']
    assert [len(r['differences']) for r in compiled['methods']]==[1,0]
    assert [(s['mode'],s['passed'],s['skipped']) for s in tests['suites']]==[('512',27,0),('256',27,0),('scalar',2,0)]
    assert not census['release_admitted'] and not census['application_scored']
    for mode in census['modes']:
        result=mode['result'];assert result['owned_count']==87 and result['retained_maps']==37 and result['public_request_passed']
    assert census['failed_release_controls']==tests['failed_release_controls']
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
    assert identities['selected']['Lokad.Onnx.dll']['sha256']=='49c3a958850d3e57daa2b7e29e6bd15ff9fc4f27af098d8ce44d8f20b9e065e8'
    consumers={name:pin(CURRENT/'collected/runtimes/candidate'/(name+'.dll')) for name in ['TranscribeReplay','AudioBenchmark']}
    assert consumers==read(CURRENT/'analysis.json')['consumers']
    copy(TOOLS/'README.md',bundle/'prospective-models.md')
    copy(ROOT/'.agent/m76-parakeet-owned-packed-weights-20260925.md',bundle/'prospective-plan.md')
    save(bundle/'stage.json',dict(passed=True,identities=identities,consumers=consumers,terminal_sources=terminals,
        failed_release_controls=read(CENSUS/'analysis.json')['failed_release_controls'],release_admitted=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'),str(path))
            originals[path.relative_to(ROOT).as_posix()]=pin(path)
    for name in ['run.py','audit.py',*UNCHANGED]:originals[(ORIGINAL/name).relative_to(ROOT).as_posix()]=pin(ORIGINAL/name)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    originals.pop('.agent/m76-parakeet-owned-packed-weights-20260925.md')
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),identities=identities)))


if __name__=='__main__':prepare()
