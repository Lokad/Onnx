"""Freeze one diagnostic only after the actual rational-sigmoid root is qualified."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from fixture import ROOT, BASE as MODELS, CORE, DATA, inspect
from protocol import TOOLS, PARENT, pin, read, save

BASE = ROOT/'artifacts/parakeet-decoder-projection-observation-v2-amd-20260927'
FAILED = ROOT/'artifacts/parakeet-decoder-projection-observation-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-rational-sigmoid-root-amd-20260927'
EVENTS = ROOT/'artifacts/parakeet-dispatch-events-amd-20260923'
EXPORT = ROOT/'artifacts/parakeet-dispatch-full-export-amd-20260923'
TRACER = ROOT/'artifacts/parakeet-current-profile-amd-20260923/payload'
REMOTE_ROOT = '/dev/shm/lokad-parakeet-rational-sigmoid-root-20260927'
REMOTE_EVENTS = '/dev/shm/lokad-parakeet-dispatch-events-20260923'
# These exact qualified exporter bytes were restored into the terminal first attempt.
REMOTE_EXPORT = '/dev/shm/lokad-parakeet-decoder-projection-observation-20260927'


def root_binding(value):
    assert value['passed'] and value['root_source_verified'] and value['consumer']['passed']
    assert value['measured']['Lokad.Onnx.dll']['sha256'] == CORE
    assert value['measured']['Lokad.Onnx.Data.dll']['sha256'] == DATA
    assert value['inventory'] == dict(passed=True, core_methods=3283, data_methods=697,
        public_surface_equal=True, assembly_attributes_equal=True, method_bodies_equal=True,
        implementation_flags_equal=True)
    assert set(value['built']) == {'Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'}
    for identity in value['built'].values():
        assert type(identity['bytes']) is int and identity['bytes'] > 0
        assert len(identity['sha256']) == 64 and all(c in '0123456789abcdef' for c in identity['sha256'])
    return value['built']


def previous_closed():
    assert (QUALIFIED/'closed.json').exists(), 'Finish actual root/package qualification first'
    proof = read(QUALIFIED/'closed.json')
    assert proof['passed'] and proof['remote_terminal']
    assert proof['analysis'] == pin(QUALIFIED/'analysis.json')
    for name, wanted in proof['files'].items(): assert pin(QUALIFIED/name) == wanted, name
    product = root_binding(read(QUALIFIED/'analysis.json'))
    applied = read(QUALIFIED/'bundle/evidence/root-applied.json')
    assert applied['passed'] and len(applied['source_files']) == 439
    assert read(QUALIFIED/'analysis.json')['root_integration'] == pin(QUALIFIED/'collected/evidence/root-applied.json')
    for name, wanted in applied['source_files'].items(): assert pin(ROOT/name) == wanted, name
    prefixes = ['src/', 'tests/Lokad.Onnx.Backend.Tests/', 'tests/Lokad.Onnx.Tensors.Tests/']
    actual = {p.relative_to(ROOT).as_posix() for prefix in prefixes for p in (ROOT/prefix).rglob('*')
              if p.is_file() and not {'bin', 'obj'} & set(p.relative_to(ROOT).parts)}
    assert actual == {n for n in applied['source_files'] if any(n.startswith(prefix) for prefix in prefixes)}
    for name, wanted in product.items(): assert pin(QUALIFIED/'collected/runtime'/name) == wanted
    for folder, digest in [(EVENTS, 'c6e1e4d42d3f377f77382a4091ad1150c1a62335a72c7d63a38c728d7a753e19'),
                           (EXPORT, '35b18e874e1e0c47a7b9e1fe6f2608b0940c1eb431c289c32ba05855cb2b5afa')]:
        assert pin(folder/'closed.json')['sha256'] == digest and read(folder/'closed.json')['passed']
    assert pin(FAILED/'failed.json')['sha256'] == '675df06cfc14b8ddc3ba01387209194d91c7616617c931ee2ecc6a14faccfa57'
    failed = read(FAILED/'failed.json')
    assert not failed['passed'] and failed['evidence_verified'] and failed['terminal'] and not failed['diagnostic_executed']
    for name, wanted in failed['files'].items(): assert pin(FAILED/name) == wanted, name
    original = read(EVENTS/'prepared.json')['files']
    for name in ['protocol.py', 'remote.py', 'run.py']:
        path = PARENT/name
        assert pin(path) == original[path.relative_to(ROOT).as_posix()], name
    return product


def prepare():
    assert not BASE.exists()
    product = previous_closed()
    spec, arrays = inspect()
    spec['core_sha256'] = product['Lokad.Onnx.dll']['sha256']
    spec['root_qualification'] = pin(QUALIFIED/'closed.json')
    model = ROOT/'models/parakeet-tdt-0.6b-v3/decoder_joint-model.onnx'
    assert pin(model)['sha256'] == spec['model_sha256']
    # No mutation until every required source and qualification exists.
    required = ['Driver.cs', 'Observer.csproj', 'protocol.py', 'remote.py', 'remote_prepare.py',
                'checks.py', 'events.py', 'run.py', 'audit.py', 'README.md']
    assert all((TOOLS/name).is_file() for name in required), 'Complete the transport/auditor before preparation'
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    originals, links = {}, {}

    def copy(source, name):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target); originals[source.relative_to(ROOT).as_posix()] = pin(source)

    def link(source, remote, target, wanted):
        assert pin(source) == wanted
        links[target] = dict(source=remote, identity=wanted)
        originals[source.relative_to(ROOT).as_posix()] = wanted

    copy(ROOT/'global.json', 'source/global.json')
    for name in ['Driver.cs', 'Observer.csproj']: copy(TOOLS/name, 'source/observer/'+name)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py', 'events.py']: copy(TOOLS/name, 'tools/'+name)
    for name in ['protocol.py', 'remote.py']: copy(PARENT/name, 'tools/'+name.replace('.py', '_base.py'))
    originals[(PARENT/'run.py').relative_to(ROOT).as_posix()] = pin(PARENT/'run.py')
    copy(TOOLS/'README.md', 'prospective-observation.md')
    copy(FAILED/'failed.json', 'evidence/first-attempt-failed.json')
    copy(FAILED/'frozen-tools.json', 'evidence/first-attempt-tools.json')
    copy(FAILED/'collected/logs/observer-build.stdout', 'evidence/first-attempt-build.stdout')
    for name, raw in arrays.items():
        path = bundle/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    save(bundle/'observation.json', spec)
    for label, folder in [('root', QUALIFIED), ('events', EVENTS), ('export', EXPORT)]:
        for name in ['closed.json', 'analysis.json', 'payload.json']: copy(folder/name, 'evidence/'+label+'/'+name)
        copy(folder/'collected/collection.json', 'evidence/'+label+'/collection.json')
    copy(QUALIFIED/'collected/evidence/root-applied.json', 'evidence/root/applied.json')
    for name, wanted in spec['provenance']['files'].items():
        assert pin(MODELS/name) == wanted; originals[(MODELS/name).relative_to(ROOT).as_posix()] = wanted
    copy(MODELS/'closed.json', 'evidence/models-closed.json')
    for name in ['Lokad.Onnx.dll', 'Google.Protobuf.dll']:
        source = QUALIFIED/'collected/runtime'/name
        wanted = read(QUALIFIED/'closed.json')['files']['collected/runtime/'+name]
        link(source, REMOTE_ROOT+'/runtime/'+name, 'runtime/'+name, wanted)
    tracer = read(TRACER/'payload.json')
    for name, wanted in tracer['files'].items():
        if name.startswith('tracer/'):
            link(TRACER/name, REMOTE_EVENTS+'/'+name, name, wanted)
    exported = read(EXPORT/'closed.json')
    for name, wanted in exported['files'].items():
        if name.startswith('collected/export-runtime/'):
            target = name.removeprefix('collected/')
            link(EXPORT/name, REMOTE_EXPORT+'/'+target, target, wanted)
    assert 'tracer/dotnet-trace.dll' in links and 'export-runtime/DispatchEventsExport.dll' in links
    originals[model.relative_to(ROOT).as_posix()] = pin(model)
    stage = dict(passed=True, product=product, links=links, model=pin(model), model_path=spec['model'],
        root_qualification=pin(QUALIFIED/'closed.json'),
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json', stage)
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix == '.py': ast.parse(path.read_text(encoding='utf8'), str(path))
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(passed=True, archive=pin(BASE/'payload.tar.gz'), links=len(links), fixture_bytes=sum(map(len, arrays.values())))))


if __name__ == '__main__': prepare()
