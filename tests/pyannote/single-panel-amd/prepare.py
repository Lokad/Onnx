"""Freeze qualified direct-output source and fresh AMD application inputs."""
import ast
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from candidate_protocol import LIMITS, pin, read, write, verified_files
from transport import ROOT, BASE, PREPARED, TOOLS, REMOTE, SITE, local_e5_terminal
sys.path.insert(0, str(SITE))
from fresh_qualification import IDENTITIES
from admission import ROLE_LABELS

OLD = ROOT / 'artifacts/pyannote-portable-amd-payload-20260922/payload'
MODEL = ROOT / 'artifacts/pyannote-single-panel-composition-20260922'
GRAPH = ROOT / 'artifacts/pyannote-single-panel-models-20260922'
SHARED = ROOT / 'artifacts/pyannote-single-panel-shared-20260922'


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    assert not PREPARED.exists() and not BASE.exists()
    local_e5_terminal()
    sys.path.insert(0, str(SITE)); import psutil
    assert psutil.virtual_memory().available >= 8 * 1024**3
    previous = ROOT / 'artifacts/pyannote-direct-amd-execution-v2-20260922'
    closed = read(previous / 'closed.json')
    assert pin(previous / 'closed.json')['sha256'] == 'ee5f73d8c286728316fe14ef7032ef42fa8fcd89a51be467f954303cfb595537'
    assert closed['passed'] and not read(previous / 'analysis.json')['performance']['admitted']
    verified_files(previous, closed['files'])
    controller = read(previous / 'controller/state.json')
    assert controller['complete'] and controller['code'] == 0
    for identity in [controller['supervisor']] + [r['child'] for r in controller['stages']]:
        try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
        except psutil.NoSuchProcess: pass
    evidence = {(previous / 'closed.json').relative_to(ROOT).as_posix(): pin(previous / 'closed.json')}
    for folder, sha, identity_key in [
        (MODEL, 'c0de18c73357515e34f1e80b112b87c9efb133e0464cccd2776874cc4610378d', 'identities'),
        (GRAPH, '56907aea97ef3af546336e61ef713c9ce5d8179d847a8363bbabfdb1463b531c', 'terminal_identities'),
        (SHARED, '72e3412c01eac72e800dbe01ba23db8a7d289ed0112b754584976c9f4dc85ea4', 'terminal_identities')]:
        path = folder / 'closed.json'; assert pin(path)['sha256'] == sha
        proof = read(path); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(ROOT / name) == wanted, name
        for identity in proof[identity_key]:
            try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
            except psutil.NoSuchProcess: pass
        evidence[path.relative_to(ROOT).as_posix()] = pin(path)
    assert read(GRAPH / 'complete.json')['exact_predecessor_public_requests'] == 16
    assert not any(r['changed_bits'] for r in read(SHARED / 'analysis.json')['rows'])
    old = read(OLD / 'payload.json'); verified_files(OLD, old['files'])
    PREPARED.mkdir(); BASE.mkdir(); payload = PREPARED / 'payload'; payload.mkdir()
    for name in old['files']:
        if name.startswith(('source/', 'runtimes/', 'graph-consumer/', 'manifests/', 'retained/')) or name == 'prospective-plan.md': continue
        copy(OLD / name, payload / name)
    shutil.copytree(MODEL / 'source', payload / 'source', ignore=shutil.ignore_patterns('bin', 'obj'))
    (payload / 'manifests').mkdir()
    for role in IDENTITIES:
        source = OLD / 'runtimes/portable' if role == 'production' else GRAPH / 'runtime'
        shutil.copytree(source, payload / 'runtimes' / role)
        for runner in ['AudioBenchmark', 'NaturalMeetings', 'TranscribeReplay']:
            for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
                copy(OLD / 'runtimes/portable' / (runner + '.' + suffix), payload / 'runtimes' / role / (runner + '.' + suffix))
        for name, sha in zip(['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'], IDENTITIES[role], strict=True):
            assert pin(payload / 'runtimes' / role / name)['sha256'] == sha
        for family in ['pyannote', 'parakeet']:
            manifest = read(OLD / 'manifests' / ('portable-' + family + '.json'))
            manifest.update(core_sha256=IDENTITIES[role][0], data_sha256=IDENTITIES[role][1])
            write(payload / 'manifests' / (role + '-' + family + '.json'), manifest)
    shutil.copytree(MODEL / 'caller/bin/Release/net10.0', payload / 'caller')
    copy(ROOT / 'artifacts/pyannote-two-column-20260922/payload/shapes.json', payload / 'caller/shapes.json')
    # These sources explain the changed consumer and caller; executed binaries
    # retain the exact independently qualified Windows identities.
    shutil.copytree(GRAPH / 'consumer', payload / 'graph-consumer', ignore=shutil.ignore_patterns('bin', 'obj'))
    copy(MODEL / 'caller/Program.cs', payload / 'caller-source/Program.cs')
    # The old copied meeting manifest names accepted production. Replace only
    # the identity literals before freezing this new, unexecuted payload.
    meeting = payload / 'meetings/manifest.json'; manifest = read(meeting)
    manifest.update(core_sha256=IDENTITIES['portable'][0], data_sha256=IDENTITIES['portable'][1])
    meeting.write_text(__import__('json').dumps(manifest, indent=2) + '\n', encoding='utf8')
    copy(TOOLS / 'README.md', payload / 'prospective-plan.md')
    spec = dict(old)
    files = {p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()}
    spec.update(source='521e41f6 plus isolated single-panel direct-output source', files=files, remote=REMOTE, role_labels=ROLE_LABELS,
                cores={r: pin(payload / 'runtimes' / r / 'Lokad.Onnx.dll') for r in IDENTITIES},
                source_files={p.relative_to(payload / 'source').as_posix(): pin(p) for p in (payload / 'source').rglob('*') if p.is_file()},
                windows_qualification=evidence,
                scope='Fresh exact-runtime AMD callers, complete models, long meetings, normal Linux source/suites and matched production/candidate/ORT comparison.')
    spec.pop('retained_qualification', None)
    spec['protocol'] = dict(old['protocol'], qualification_reused=False, caller_cases_per_mode=400,
                            performance_admission=pin(TOOLS / 'admission.py'))
    spec['graph_consumer_adaptation'] = read(GRAPH / 'prepared.json')['changed_literal']
    write(payload / 'payload.json', spec)
    with tarfile.open(PREPARED / 'payload.tar.gz', 'w:gz') as archive:
        for name in [*sorted(files), 'payload.json']: archive.add(payload / name, arcname=name, recursive=False)
    prepared = dict(passed=True, payload=pin(payload / 'payload.json'), archive=pin(PREPARED / 'payload.tar.gz'),
                    files=len(files), bytes=sum(v['bytes'] for v in files.values()), scope=spec['scope'])
    write(PREPARED / 'prepared.json', prepared)
    target = BASE / 'execution'; target.mkdir(); local_tools = dict(evidence)
    for p in sorted(TOOLS.glob('*.py')):
        ast.parse(p.read_text(encoding='utf8'), filename=str(p))
        local_tools[p.relative_to(ROOT).as_posix()] = pin(p); copy(p, target / p.name)
    original = ROOT / 'tests/parakeet/transcribe/audit.py'
    local_tools[original.relative_to(ROOT).as_posix()] = pin(original); copy(original, target / 'parakeet_audit.py')
    env = dict(os.environ, PYTHONPATH=str(SITE), PYTHONDONTWRITEBYTECODE='1'); env.pop('PYTHONOPTIMIZE', None)
    with (BASE / 'selftest.log').open('x') as log:
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover', '-s', str(TOOLS), '-p', 'test_*.py', '-v'],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0, 'Selftests failed; preserve this preparation.'
    previous = read(ROOT / 'artifacts/pyannote-portable-amd-execution-20260922/execution/execution.json')
    execution = dict(schema=1, source=spec['source'], payload=prepared['payload'], payload_archive=prepared['archive'],
                     remote=REMOTE, limits=LIMITS, local_tools=local_tools,
                     files={p.name: pin(p) for p in target.iterdir() if p.is_file()}, external=previous['external'],
                     selftest=pin(BASE / 'selftest.log'), scope=spec['scope'])
    write(target / 'execution.json', execution)
    with tarfile.open(BASE / 'execution.tar.gz', 'w:gz') as archive:
        for p in sorted(target.iterdir()): archive.add(p, arcname=p.name, recursive=False)
    bundle = dict(passed=True, execution=pin(target / 'execution.json'), archive=pin(BASE / 'execution.tar.gz'),
                  payload_archive=prepared['archive'], files=len(execution['files']), selftest=execution['selftest'],
                  amd_deployed=False, amd_qualified=False, amd_timing=False)
    write(BASE / 'prepared.json', bundle)
    print(dict(payload=prepared, execution=bundle))


if __name__ == '__main__': main()
