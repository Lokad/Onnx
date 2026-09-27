"""Freeze a single isolated source candidate and its real projection fixture."""
import ast
import difflib
import json
from pathlib import Path
import tarfile
from protocol import TOOLS, PARENT, pin, read, save
from source import TARGET, HELPER, changed

ROOT = TOOLS.parents[2]
BASE = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-v2-amd-20260927'
FIRST = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-rational-sigmoid-root-amd-20260927'
OBSERVATION = ROOT/'artifacts/parakeet-decoder-projection-observation-v3-amd-20260927'
REMOTE_ROOT = '/dev/shm/lokad-parakeet-rational-sigmoid-root-20260927'
REMOTE_FIRST = '/dev/shm/lokad-decrow-20260927'


def previous_closed():
    for folder, digest in [(QUALIFIED, 'f6df53ce2ab773898bc84f144abeda97908d383fef9a4df4b7299a22a4d3594d'),
            (OBSERVATION, 'ef94387e0517b08fd87d010c08d99e0b801e1bd7d2809c04b1dcec614333397d')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        closed = read(folder/'closed.json'); assert closed['passed'] and closed['remote_terminal']
        for name, wanted in closed['files'].items(): assert pin(folder/name) == wanted, name
    applied = read(QUALIFIED/'bundle/evidence/root-applied.json')
    assert applied['passed'] and len(applied['source_files']) == 439
    for name, wanted in applied['source_files'].items(): assert pin(ROOT/name) == wanted, name
    original_tools = read(OBSERVATION/'prepared.json')['files']
    for path in [PARENT/'protocol.py', PARENT/'remote.py', PARENT/'run.py', TOOLS.parent/'decoder-projection-observation/run.py']:
        assert pin(path) == original_tools[path.relative_to(ROOT).as_posix()], path
    assert pin(FIRST/'failed.json')['sha256'] == '6225233c00528cde3170a535ac973ddc7382c3af5086b370d90ff860d023d24d'
    failed = read(FIRST/'failed.json')
    assert failed['evidence_verified'] and failed['terminal'] and failed['builds_passed']
    assert not failed['inventory_executed'] and not failed['contracts_executed']
    for name, wanted in failed['files'].items(): assert pin(FIRST/name) == wanted, name
    for name in ['Contracts.cs.txt', 'Contracts.csproj', 'PreparedSingleRowKernel.cs.txt', 'source.py']:
        assert pin(TOOLS/name) == pin(FIRST/'frozen-tools'/name), 'Keep the product and consumer unchanged'
    return applied['source_files']


def prepare():
    assert not BASE.exists()
    sources = previous_closed()
    required = ['Contracts.cs.txt', 'Contracts.csproj', 'PreparedSingleRowKernel.cs.txt',
        'audit.py', 'README.md', 'remote_prepare.py', 'remote.py', 'commands.py', 'compiled.py', 'execmode.py']
    assert all((TOOLS/name).is_file() for name in required)
    for path in TOOLS.glob('*.py'): ast.parse(path.read_text(encoding='utf8'), str(path))
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    original = {name: wanted for name, wanted in sources.items()}

    def put(name, data):
        path = bundle/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)

    def copy(path, target):
        original[path.relative_to(ROOT).as_posix()] = pin(path)
        put(target, path.read_bytes())

    for name in sources: put('source/'+name, (ROOT/name).read_bytes())
    before = (ROOT/TARGET).read_text(encoding='utf8')
    after = changed(before)
    put('source/'+TARGET, after.encode())
    put('source/'+HELPER, (TOOLS/'PreparedSingleRowKernel.cs.txt').read_bytes())
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=TARGET, tofile=TARGET))
    put('candidate.patch', patch.encode())
    copy(TOOLS/'Contracts.cs.txt', 'source/contracts/Contracts.cs')
    copy(TOOLS/'Contracts.csproj', 'source/contracts/Contracts.csproj')
    for name in ['protocol.py', 'remote.py', 'commands.py', 'compiled.py', 'remote_prepare.py', 'execmode.py']:
        copy(TOOLS/name, 'tools/'+name)
    for name in ['protocol', 'remote']: copy(PARENT/(name+'.py'), 'tools/'+name+'_base.py')
    copy(TOOLS/'README.md', 'prospective-contracts.md')
    copy(ROOT/'PLAN.md', 'prospective-plan.md')
    # The plan is frozen in the bundle; its living root copy may continue to change.
    original.pop('PLAN.md')
    for label, folder in [('root', QUALIFIED), ('observation', OBSERVATION)]:
        for name in ['closed.json', 'analysis.json']:
            if label == 'observation' and name == 'analysis.json': continue
            copy(folder/name, 'evidence/'+label+'/'+name)
        copy(folder/'collected/collection.json', 'evidence/'+label+'/collection.json')
    copy(FIRST/'failed.json', 'evidence/first-failed.json')
    copy(FIRST/'collected/collection.json', 'evidence/first-collection.json')
    copy(FIRST/'collected/built.json', 'built.json')
    copy(QUALIFIED/'bundle/evidence/root-applied.json', 'evidence/root-applied.json')
    value = read(OBSERVATION/'analysis.json')
    model = read(OBSERVATION/'collected/observation.json')
    copy(OBSERVATION/'collected/control-run/projection-a.f32', 'projection-a.f32')
    fixture = dict(model=model['model'], model_sha256=model['model_sha256'],
        a_sha256=value['operands']['a']['sha256'], b_sha256=value['operands']['b']['sha256'],
        packed_sha256=value['mapping']['entry']['packed']['sha256'],
        output_sha256=value['operands']['output']['sha256'])
    save(bundle/'fixture.json', fixture)
    links = {}
    first_collection = read(FIRST/'collected/collection.json')

    def link(name, source):
        local = FIRST/'collected'/source
        wanted = first_collection['files'][source]
        assert pin(local) == wanted
        original[local.relative_to(ROOT).as_posix()] = wanted
        links[name] = dict(source=REMOTE_FIRST+'/'+source, identity=wanted)

    for name in read(FIRST/'collected/built.json')['files']: link(name, name)
    for suffix in ['dll', 'deps.json', 'runtimeconfig.json']: link('bridge/Bridge.'+suffix, 'bridge/Bridge.'+suffix)
    stage = dict(passed=True, links=links, current_product=value['product'], model=pin(ROOT/'models/parakeet-tdt-0.6b-v3/decoder_joint-model.onnx'),
        model_path=fixture['model'], changed_methods=['ResolvePackedKernel', 'RunPreparedPackedRows'], added_method='PreparedSingleRowKernel.Multiply',
        source={name: pin(bundle/'source'/name) for name in [*sources, HELPER]},
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert stage['model']['sha256'] == fixture['model_sha256']
    assert stage['source'] == read(FIRST/'bundle/stage.json')['source']
    save(bundle/'stage.json', stage)
    for path in TOOLS.iterdir():
        if path.is_file(): original[path.relative_to(ROOT).as_posix()] = pin(path)
    for path in [PARENT/'run.py', TOOLS.parent/'decoder-projection-observation/run.py']:
        original[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=original, archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(passed=True, source_files=len(stage['source']), links=len(links), archive=pin(BASE/'payload.tar.gz'))))


if __name__ == '__main__': prepare()
