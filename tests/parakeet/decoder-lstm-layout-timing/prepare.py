"""Freeze one qualified-contract candidate; reuse actual calls and timing consumer."""
import ast
import json
from pathlib import Path
import tarfile
from protocol import TOOLS, PARENT, pin, read, save

ROOT = TOOLS.parents[2]
BASE = ROOT/'artifacts/parakeet-decoder-lstm-layout-timing-amd-20260927'
V3 = ROOT/'artifacts/parakeet-decoder-lstm-layout-contracts-v3-amd-20260927'
CONTROL = ROOT/'artifacts/parakeet-decoder-lstm-layout-scalar-baseline-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-packed-row-root-amd-20260927'
CALLS = ROOT/'artifacts/parakeet-prepared-recurrence-calls-amd-v2-20260924'


def previous_closed():
    for folder, name, digest in [
        (V3, 'failed.json', '59f6f738dcd71beec865e46c511a9bd151207a26d2901c259f015b387a27f63f'),
        (CONTROL, 'closed.json', '63c97822999a74921c2e8a3c64e0af9ec6682a6829ba83de52bba353b93b85ec'),
        (QUALIFIED, 'closed.json', 'd0a78cdd3106d6a72a41303f879bbb2f9ea3bd6a298d778333015f38ccdac246'),
        (CALLS, 'closed.json', '20b3bc4b5a2d2ddb6e0834e40fb7e2bcb5220dd1f45c813a7a830466ddbc463d')]:
        assert pin(folder/name)['sha256'] == digest
        for filename, wanted in read(folder/name)['files'].items(): assert pin(folder/filename) == wanted, filename
    diagnosis = read(CONTROL/'analysis.json')
    assert not diagnosis['contract_regression_found'] and not diagnosis['original_campaign_passed']
    assert diagnosis['matched_existing_scalar_cases'] == 172 and diagnosis['matched_scalar_rejections'] == 18
    assert diagnosis['projection_hashes_equal_across_modes'] and diagnosis['total_projection_values'] == 5836800
    for name, wanted in read(QUALIFIED/'bundle/evidence/root-applied.json')['source_files'].items():
        assert pin(ROOT/name) == wanted, name


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def put(name, data):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    def copy(source, name):
        originals[source.relative_to(ROOT).as_posix()] = pin(source); put(name, source.read_bytes())
    copy(ROOT/'global.json', 'source/global.json')
    copy(PARENT/'Timing.csproj', 'source/Timing.csproj')
    timing = (PARENT/'Timing.cs').read_text()
    originals[(PARENT/'Timing.cs').relative_to(ROOT).as_posix()] = pin(PARENT/'Timing.cs')
    changes = {
        '(role is "selected" or "candidate")': '(role is "selected" or "candidate" or "selectedfallback" or "candidatefallback")',
        'var graph = new ComputationalGraph(Budget);': 'long budget = role.EndsWith("fallback") ? 0 : Budget;\n            var graph = new ComputationalGraph(budget);',
        '(role == "candidate" ? 13107200 : 0)': '(role.EndsWith("fallback") ? 0 : 13107200)',
        'graph.MaximumPackedWeightBytes == Budget': 'graph.MaximumPackedWeightBytes == budget',
    }
    for before, after in changes.items():
        assert timing.count(before) == 1, before; timing = timing.replace(before, after)
    put('source/Timing.cs', timing.encode())
    for name in ['protocol', 'checks', 'remote']:
        copy(TOOLS/(name+'.py'), 'tools/'+name+'.py')
        copy(PARENT/(name+'.py'), 'tools/'+name+'_base.py')
    copy(PARENT/'campaign_processes.py', 'tools/campaign_processes.py')
    copy(TOOLS/'remote_prepare.py', 'tools/remote_prepare.py')
    copy(TOOLS/'README.md', 'prospective-timing.md')
    put('prospective-plan.md', (ROOT/'PLAN.md').read_bytes())
    for label, folder, closure in [('candidate', V3, 'failed.json'), ('control', CONTROL, 'closed.json'),
        ('root', QUALIFIED, 'closed.json'), ('calls', CALLS, 'closed.json')]:
        copy(folder/closure, 'evidence/'+label+'/'+closure)
        copy(folder/'collected/collection.json', 'evidence/'+label+'/collection.json')
    copy(CONTROL/'analysis.json', 'evidence/contracts-diagnosis.json')
    identities = {}
    for role in ['selected', 'candidate', 'selectedfallback', 'candidatefallback']:
        original = 'current' if role.startswith('selected') else 'candidate'
        identities[role] = {}
        for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll', 'Google.Protobuf.dll']:
            source = V3/'collected/runtimes'/original/name
            copy(source, 'products/'+role+'/'+name); identities[role][name] = pin(source)
    assert identities['selected']['Lokad.Onnx.dll']['sha256'] == '0d224bcff591563816d64c3a3cc51f7b7cd38f5a1d3c523b05c699a9b82b6b97'
    assert identities['candidate']['Lokad.Onnx.dll']['sha256'] == 'ad97b4ad632b3306ea3a39d14549ed98540bc8c2e85953a68d89b453d3ed2fdc'
    save(bundle/'spec.json', dict(identities=identities))
    links = {}
    for source in (CALLS/'collected/fixtures').iterdir():
        assert source.is_file(); wanted = pin(source)
        originals[source.relative_to(ROOT).as_posix()] = wanted
        links['fixtures/'+source.name] = dict(source='fixtures/'+source.name, pin=wanted)
    assert len(links) == 444
    cases = {}
    for call in read(CALLS/'collected/fixtures/result.json')['calls']:
        cases[call['name']] = cases.get(call['name'], 0)+1
    assert list(cases.values()) == [74, 58, 92, 8, 74, 74]
    save(bundle/'stage.json', dict(passed=True, identities=identities, links=links, cases=cases,
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for source in [*TOOLS.iterdir(), PARENT/'run.py', PARENT/'audit.py']:
        if source.is_file():
            if source.suffix == '.py': ast.parse(source.read_text(), str(source))
            originals[source.relative_to(ROOT).as_posix()] = pin(source)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, stage=pin(bundle/'stage.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(passed=True, archive=pin(BASE/'payload.tar.gz'), captured_files=len(links), clocks=60800)))


if __name__ == '__main__': prepare()
