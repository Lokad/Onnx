"""Reuse full Parakeet correctness after this candidate passes graph admission."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'observed-dense-where-models-amd'
sys.path.insert(1, str(PARENT))
from protocol import pin, read, save

BASE = ROOT/'artifacts/parakeet-owned-batch-isolation-models-amd-20260925'
CURRENT = ROOT/'artifacts/parakeet-direct-depthwise-models-amd-20260925'
PREVIOUS = CURRENT
BUILD = ROOT/'artifacts/parakeet-owned-batch-isolation-build-amd-20260925'
GRAPHS = ROOT/'artifacts/parakeet-owned-batch-graph-qualification-20260925'
ORIGINAL_GRAPHS = ROOT/'artifacts/parakeet-owned-batch-isolation-graphs-amd-20260925'
SHORT_E5 = ROOT/'artifacts/e5-steady-short-amd-20260925'
GRAPH_QUALIFIER = ROOT/'tests/benchmarks/e5-steady-short-results/qualified_graphs.py'
SOURCE = ROOT/'artifacts/parakeet-owned-batch-isolation-source-20260925'
REMOTE_CURRENT = '/dev/shm/lokad-parakeet-direct-depthwise-models-20260925'
REMOTE_BUILD = '/dev/shm/lokad-parakeet-owned-batch-isolation-build-20260925'
REMOTE_GRAPHS = '/dev/shm/lokad-parakeet-owned-batch-isolation-graphs-20260925'
REMOTE_SHORT_E5 = '/dev/shm/lokad-e5-steady-short-20260925'
UNCHANGED = ['protocol.py','remote.py','checks.py','native_audit.py','public_audit.py']
LABELS = dict(selected='Direct-depthwise parent', candidate='Packed-dispatch isolation')


def previous_closed():
    # No preparation or VM work can precede the complete graph verdict.
    loader = importlib.util.spec_from_file_location('isolation_graph_qualification', GRAPH_QUALIFIER)
    qualifier = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(qualifier)
    graph = qualifier.admission()
    assert graph['passed'] and graph['admitted'] and graph['all_controls_passed']
    assert graph['original_graph_failure_preserved']
    for folder, digest in [(CURRENT,'1dd7208779450c5ede5da6900614abc7a521c8d8eacedf6c11ba7976b432bd2b'),
                           (BUILD,'5dd53e90d4924a36bb6f43cc80fa939e9236492949542dcf659dc056d4ea3860'),
                           (GRAPHS,None)]:
        if digest is not None: assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    build = read(BUILD/'analysis.json'); prior = read(CURRENT/'analysis.json')
    assert graph['products']['candidate']['Lokad.Onnx.dll'] == build['product']['Lokad.Onnx.dll']
    assert build['product']['Lokad.Onnx.dll']['sha256'] == 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
    assert prior['identities']['candidate']['Lokad.Onnx.dll']['sha256'] == '40260aef7fd93c5153601ec104a87843a2c017a2720fd64c9e24e3460d455749'
    assert build['product']['Lokad.Onnx.Data.dll'] == prior['identities']['candidate']['Lokad.Onnx.Data.dll']
    assert build['source'] == pin(SOURCE/'prepared.json')
    assert read(BUILD/'build-review.json')['release_dispatcher_restored']
    assert [s['passed'] for s in build['suites']] == [64,25]
    expected = (PARENT/'audit.py').read_text().replace('M70 current release', LABELS['selected']).replace('M70 observed-mask composition', LABELS['candidate'])
    assert (TOOLS/'audit.py').read_text() == expected
    return build, prior


def prepare():
    build, prior = previous_closed(); assert not BASE.exists()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source, name):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target); originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in UNCHANGED: copy(PARENT/name, 'tools/'+name)
    copy(TOOLS/'remote_prepare.py', 'tools/remote_prepare.py')
    copy(TOOLS/'README.md', 'prospective-models.md')
    shutil.copy2(ROOT/'PLAN.md', bundle/'prospective-plan.md')
    old = read(CURRENT/'payload.json'); old_receipt = read(CURRENT/'collected/collection.json')
    links = {}
    for name, wanted in old['files'].items():
        if name.startswith(('assets/','parakeet-reference/')):
            assert pin(CURRENT/'collected'/name) == wanted == old_receipt['files'][name]
            links[name] = dict(source=REMOTE_CURRENT+'/'+name, identity=wanted)
    identities = dict(selected=prior['identities']['candidate'], candidate=build['product'])
    for role in ['selected','candidate']:
        for name, wanted in old['files'].items():
            if name.startswith('runtimes/candidate/'):
                suffix = name.removeprefix('runtimes/candidate/')
                links['runtimes/'+role+'/'+suffix] = dict(source=REMOTE_CURRENT+'/'+name, identity=wanted)
        for name, wanted in identities[role].items():
            remote = REMOTE_BUILD+'/runtime/' if role == 'candidate' else REMOTE_CURRENT+'/runtimes/candidate/'
            links['runtimes/'+role+'/'+name] = dict(source=remote+name, identity=wanted)
    terminals = []
    for folder, label, remote, receipt in [(CURRENT,'parent',REMOTE_CURRENT,'collected/collection.json'),
                                         (BUILD,'build',REMOTE_BUILD,'capture-collected/capture-collection.json'),
                                         (ORIGINAL_GRAPHS,'original-graphs',REMOTE_GRAPHS,'collected/collection.json'),
                                         (SHORT_E5,'short-e5',REMOTE_SHORT_E5,'collected/collection.json')]:
        for name in ['closed.json','analysis.json']: copy(folder/name, 'evidence/'+label+'-'+name)
        target = 'evidence/'+label+'-collection.json'; copy(folder/receipt, target)
        terminals.append(dict(remote=remote+'/'+Path(receipt).name, local=target))
    for name in ['closed.json','analysis.json']:
        copy(GRAPHS/name, 'evidence/graphs-'+name)
    copy(GRAPH_QUALIFIER, 'evidence/graphs-generator.py')
    copy(CURRENT/'collected/manifests/candidate-parakeet.json', 'evidence/original-manifest.json')
    copy(BUILD/'build-review.json', 'evidence/build-review.json')
    stage = dict(passed=True, identities=identities, consumers=prior['consumers'], links=links,
                 labels=LABELS, terminals=terminals, external=old['external'], interpreter=old['interpreter'],
                 release_admitted=False, failed_release_controls=[],
                 previous_owner=read(SHORT_E5/'collected/collection.json')['identities'][0],
                 files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json', stage)
    for folder in [TOOLS,PARENT]:
        for path in folder.iterdir():
            if path.is_file():
                if path.suffix == '.py': ast.parse(path.read_text(), str(path))
                originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),identities=identities)))


if __name__ == '__main__':
    prepare()
