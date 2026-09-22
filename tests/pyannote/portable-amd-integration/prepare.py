"""Freeze exact portable source, retained qualification and fresh production trial."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from candidate_protocol import LIMITS, pin, read, write, verified_files
from admission import ROLE_LABELS
from retained_qualification import IDENTITIES, CLOSURES, qualification
from transport import ROOT, BASE, PREPARED, TOOLS, REMOTE, SITE, local_e5_terminal

OLD = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload'
PORTABLE = ROOT / 'artifacts/pyannote-portable-applications-20260922'
SOURCE = ROOT / 'artifacts/pyannote-portable-integration-tests-20260922/source'
QUALIFIED = dict(production=ROOT / 'artifacts/pyannote-amd-execution-v4-20260921',
    portable=ROOT / 'artifacts/pyannote-combined-amd-execution-v2-20260922')
PAYLOADS = dict(production=OLD,
    portable=ROOT / 'artifacts/pyannote-combined-amd-payload-v2-20260922/payload')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    assert not PREPARED.exists() and not BASE.exists()
    local_e5_terminal()
    sys.path.insert(0, str(SITE))
    import psutil
    assert psutil.virtual_memory().available >= 8 * 1024**3
    local = ROOT / 'artifacts/pyannote-portable-integration-tests-20260922'
    assert pin(local / 'closed.json')['sha256'] == 'bf9822426cd75a86cc8c02c065005f0005e1a80b4ee63d5c53313fed145235f6'
    assert read(local / 'closed.json')['passed']
    for name, wanted in read(local / 'closed.json')['files'].items():
        assert pin(ROOT / name) == wanted, name
    for role, folder in QUALIFIED.items():
        assert pin(folder / 'closed.json')['sha256'] == CLOSURES[role]
        closed = read(folder / 'closed.json')
        assert closed['passed']
        verified_files(folder, closed['files'])
        state = read(folder / 'controller/state.json')
        assert state['complete'] and state['code'] == 0
        for identity in [state['supervisor']] + [r['child'] for r in state['stages'] if 'child' in r]:
            try:
                assert psutil.Process(identity['pid']).create_time() != identity['birth']
            except psutil.NoSuchProcess:
                pass
        verified_files(PAYLOADS[role], read(PAYLOADS[role] / 'payload.json')['files'])
    subprocess.run(['git', 'apply', '--check', 'tests/pyannote/portable-integration-tests/integration.patch'], cwd=ROOT, check=True)
    PREPARED.mkdir(); BASE.mkdir()
    payload = PREPARED / 'payload'; payload.mkdir()
    old = read(OLD / 'payload.json')
    for name in old['files']:
        if name.startswith(('source/', 'runtimes/', 'tools/', 'manifests/')) or name == 'prospective-plan.md':
            continue
        copy(OLD / name, payload / name)
    shutil.copytree(SOURCE, payload / 'source', ignore=shutil.ignore_patterns('bin', 'obj'))
    cores = {}
    for role in IDENTITIES:
        source = PAYLOADS[role]
        shutil.copytree(source / 'runtimes' / role, payload / 'runtimes' / role)
        for suffix in ('dll', 'deps.json', 'runtimeconfig.json'):
            copy(PORTABLE / 'runtime' / ('NaturalMeetings.' + suffix), payload / 'runtimes' / role / ('NaturalMeetings.' + suffix))
        cores[role] = pin(payload / 'runtimes' / role / 'Lokad.Onnx.dll')
        prior = payload / 'retained' / role
        for name in ('closed.json', 'analysis.json'):
            copy(QUALIFIED[role] / name, prior / name)
        copy(source / 'payload.json', prior / 'payload.json')
        for family in ('pyannote', 'parakeet'):
            name = role + '-' + family + '.json'
            copy(source / 'manifests' / name, payload / 'manifests' / name)
            copy(source / 'manifests' / name, prior / name)
    qualification(payload)
    meetings = payload / 'meetings'; meetings.mkdir()
    shutil.copytree(PORTABLE / 'meetings/inputs', meetings / 'inputs')
    manifest = read(PORTABLE / 'meetings/manifest.json')
    assert (manifest['core_sha256'], manifest['data_sha256']) == IDENTITIES['portable']
    copy(PORTABLE / 'meetings/manifest.json', meetings / 'manifest.json')
    copy(PORTABLE / 'meetings-run-output/result.json', meetings / 'portable-reference.json')
    native = ROOT / 'artifacts/pyannote-optimized-meetings-20260921/prior/native.json'
    assert pin(native) == read(PORTABLE / 'prepared.json')['files'][native.relative_to(ROOT).as_posix()]
    copy(native, meetings / 'native.json')
    external = dict(old['external'])
    for item in manifest['models'].values():
        name = '/home/vermorel/Onnx/' + item['amd_path']
        wanted = {k: item[k] for k in ['bytes', 'sha256']}
        assert name not in external or external[name] == wanted
        external[name] = wanted
    copy(TOOLS / 'README.md', payload / 'prospective-plan.md')
    spec = dict(old)
    files = {p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()}
    spec.update(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        files=files, cores=cores, external=external, remote=REMOTE, role_labels=ROLE_LABELS,
        source_files={p.relative_to(payload / 'source').as_posix(): pin(p) for p in (payload / 'source').rglob('*') if p.is_file()},
        retained_qualification={role: pin(folder / 'closed.json') for role, folder in QUALIFIED.items()},
        scope='Exact portable-only versus root production and fresh ORT; retained exact-runtime model/public evidence, new normal Linux suites and portable long meetings; prospective integration selection.')
    spec['protocol'] = dict(old['protocol'], natural_meetings=3,
        qualification_roles=['production', 'portable'], qualification_reused=True,
        timing_roles=['production','portable','ort','ort','portable','production'],
        total_timing_calls=96, measured_timing_calls=72, warmup_timing_calls=24,
        performance_admission=pin(TOOLS / 'admission.py'))
    spec.pop('test_harness_adaptation', None)
    write(payload / 'payload.json', spec)
    with tarfile.open(PREPARED / 'payload.tar.gz', 'w:gz') as archive:
        for name in [*sorted(files), 'payload.json']:
            archive.add(payload / name, arcname=name, recursive=False)
    prepared = dict(passed=True, payload=pin(payload / 'payload.json'), archive=pin(PREPARED / 'payload.tar.gz'),
        files=len(files), bytes=sum(v['bytes'] for v in files.values()), scope=spec['scope'])
    write(PREPARED / 'prepared.json', prepared)
    target = BASE / 'execution'
    target.mkdir()
    local_tools = {}
    for p in sorted(TOOLS.glob('*.py')):
        ast.parse(p.read_text(encoding='utf8'), filename=str(p))
        local_tools[p.relative_to(ROOT).as_posix()] = pin(p)
        copy(p, target / p.name)
    original = ROOT / 'tests/parakeet/transcribe/audit.py'
    local_tools[original.relative_to(ROOT).as_posix()] = pin(original)
    copy(original, target / 'parakeet_audit.py')
    env = dict(os.environ, PYTHONPATH=str(SITE), PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONOPTIMIZE', None)
    with (BASE / 'selftest.log').open('x') as log:
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover', '-s', str(TOOLS),
            '-p', 'test_*.py', '-v'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0, 'Selftests failed; preserve this preparation.'
    # The old execution inventory already independently verified all native dependencies.
    prior_execution = read(QUALIFIED['production'] / 'execution/execution.json')
    execution = dict(schema=1, source=spec['source'], payload=prepared['payload'], payload_archive=prepared['archive'],
        remote=REMOTE, limits=LIMITS, local_tools=local_tools, files={p.name: pin(p) for p in target.iterdir() if p.is_file()},
        external=prior_execution['external'], native_predecessor=pin(QUALIFIED['production'] / 'execution/execution.json'),
        selftest=pin(BASE / 'selftest.log'), scope=spec['scope'])
    write(target / 'execution.json', execution)
    with tarfile.open(BASE / 'execution.tar.gz', 'w:gz') as archive:
        for p in sorted(target.iterdir()):
            archive.add(p, arcname=p.name, recursive=False)
    bundle = dict(passed=True, execution=pin(target / 'execution.json'), archive=pin(BASE / 'execution.tar.gz'),
        payload_archive=prepared['archive'], files=len(execution['files']), selftest=execution['selftest'],
        amd_deployed=False, amd_qualified=False, amd_timing=False)
    write(BASE / 'prepared.json', bundle)
    print(dict(payload=prepared, execution=bundle))


if __name__ == '__main__':
    main()
