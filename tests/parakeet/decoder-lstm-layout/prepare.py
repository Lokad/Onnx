"""Freeze the qualified source plus one isolated layout change and existing captures."""
import ast
import difflib
import json
from pathlib import Path
import re
import tarfile
from protocol import TOOLS, PARENT, pin, read, save
from source import TARGETS, changed
from checks import census

ROOT = TOOLS.parents[2]
BASE = ROOT/'artifacts/parakeet-decoder-lstm-layout-contracts-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-packed-row-root-amd-20260927'
CAPTURE = ROOT/'artifacts/parakeet-prepared-recurrence-calls-amd-v2-20260924'
REMOTE_ROOT = '/dev/shm/lokad-parakeet-decoder-packed-row-root-20260927'
REMOTE_CAPTURE = '/dev/shm/lokad-parakeet-prepared-recurrence-calls-v2-20260924'
FIXTURE = 'tests/Lokad.Onnx.Backend.Tests/LstmLayoutContracts.cs'


def previous_closed():
    for folder, digest in [(QUALIFIED, 'd0a78cdd3106d6a72a41303f879bbb2f9ea3bd6a298d778333015f38ccdac246'),
        (CAPTURE, '20b3bc4b5a2d2ddb6e0834e40fb7e2bcb5220dd1f45c813a7a830466ddbc463d')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    sources = read(QUALIFIED/'bundle/evidence/root-applied.json')['source_files']
    assert len(sources) == 441 and not (ROOT/FIXTURE).exists()
    for name, wanted in sources.items(): assert pin(ROOT/name) == wanted, name
    return sources


def prepare():
    assert not BASE.exists()
    sources = previous_closed()
    before = {name: (ROOT/name).read_text(encoding='utf8') for name in TARGETS}
    after = changed(before)
    for path in TOOLS.glob('*.py'): ast.parse(path.read_text(encoding='utf8'), str(path))
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = dict(sources)
    def put(name, data):
        path = bundle/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    def copy(source, name):
        originals[source.relative_to(ROOT).as_posix()] = pin(source); put(name, source.read_bytes())
    for name in sources: put('source/'+name, after[name].encode() if name in after else (ROOT/name).read_bytes())
    copy(TOOLS/'LstmLayoutContracts.cs', 'source/'+FIXTURE)
    patch = ''.join(''.join(difflib.unified_diff(before[name].splitlines(True), after[name].splitlines(True),
        fromfile=name, tofile=name)) for name in TARGETS)
    put('candidate.patch', patch.encode())
    for name in ['protocol.py', 'commands.py', 'checks.py', 'remote.py', 'remote_prepare.py', 'testmode.py']:
        copy(TOOLS/name, 'tools/'+name)
    for name in ['protocol', 'remote']: copy(PARENT/(name+'.py'), 'tools/'+name+'_base.py')
    copy(TOOLS/'README.md', 'prospective-contracts.md')
    put('prospective-plan.md', (ROOT/'PLAN.md').read_bytes())
    for label, folder in [('root', QUALIFIED), ('capture', CAPTURE)]:
        for name in ['closed.json', 'analysis.json']: copy(folder/name, 'evidence/'+label+'/'+name)
        copy(folder/'collected/collection.json', 'evidence/'+label+'/collection.json')
    copy(QUALIFIED/'bundle/evidence/root-applied.json', 'evidence/root-applied.json')
    for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
        copy(QUALIFIED/'collected/built'/('Bridge.'+suffix), 'bridge/Bridge.'+suffix)
    current = read(QUALIFIED/'analysis.json')['built']
    for path in sorted((QUALIFIED/'collected/runtime').glob('*.dll')):
        name = path.name
        if name in current: assert pin(path) == current[name]
        copy(path, 'runtimes/current/'+name)
    expected = {name: outcome for name, outcome in census(QUALIFIED/'collected/backend-tests/backend.trx').items() if 'lstm' in name.lower()}
    alternate = {name: outcome for name, outcome in census(QUALIFIED/'collected/backend-tests-256/backend.trx').items() if 'lstm' in name.lower()}
    assert expected == alternate and len(expected) == 172 and set(expected.values()) == {'Passed'}
    tests = re.findall(r'\[Fact\]\s+public void (\w+)\(', (TOOLS/'LstmLayoutContracts.cs').read_text())
    assert len(tests) == 4
    added = ['Lokad.Onnx.Backend.Tests.LstmLayoutContracts.'+name for name in tests]
    links = {}; receipt = read(CAPTURE/'collected/collection.json')
    for path in (CAPTURE/'collected/fixtures').iterdir():
        assert path.is_file()
        name = 'fixtures/'+path.name; wanted = pin(path); assert wanted == receipt['files'][name]
        originals[path.relative_to(ROOT).as_posix()] = wanted
        links[name] = dict(source=REMOTE_CAPTURE+'/'+name, identity=wanted)
    assert len(links) == 444
    capture = read(CAPTURE/'collected/fixtures/result.json')
    assert capture['model_sha256'] == pin(ROOT/'models/parakeet-tdt-0.6b-v3/decoder_joint-model.onnx')['sha256']
    assert len(capture['calls']) == 380 and {c['index'] for c in capture['calls']} == {0, 1}
    for index in [0, 1]:
        rows = [c for c in capture['calls'] if c['index'] == index]
        assert len(rows) == 190
        assert all([c['inputs'][i] for i in [1,2,3]] == [rows[0]['inputs'][i] for i in [1,2,3]] for c in rows)
    stage = dict(passed=True, current_product=current, links=links, expected_census=expected, added_tests=added,
        source={name: pin(bundle/'source'/name) for name in [*sources, FIXTURE]},
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert {name for name, wanted in stage['source'].items() if sources.get(name) != wanted} == {*TARGETS, FIXTURE}
    save(bundle/'stage.json', stage)
    for path in TOOLS.iterdir():
        if path.is_file(): originals[path.relative_to(ROOT).as_posix()] = pin(path)
    for path in [PARENT/'run.py', TOOLS.parent/'decoder-projection-observation/run.py']:
        originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(passed=True, source_files=len(stage['source']), links=len(links), tests=len(expected)+len(added), archive=pin(BASE/'payload.tar.gz'))))


if __name__ == '__main__': prepare()
