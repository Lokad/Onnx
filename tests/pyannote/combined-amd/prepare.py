"""Freeze the qualified composition, controls, meetings and successor protocol."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from candidate_protocol import LIMITS, pin, read, write, verified_files
from admission import ROLE_LABELS
from transport import ROOT, BASE, PREPARED, TOOLS, REMOTE, SITE

OLD = ROOT / 'artifacts/pyannote-amd-candidates-v3-20260921/payload'
COMBINED = ROOT / 'artifacts/pyannote-combined-avx512-20260922'
PORTABLE = ROOT / 'artifacts/pyannote-portable-applications-20260922'


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    assert not PREPARED.exists() and not BASE.exists()
    sys.path.insert(0, str(SITE))
    import psutil
    for folder, sha in [(COMBINED, 'dd4cc7cc3bbde61e1de4b113657b72a27850bfa53dcd26c4dbeed2ef32f96e36'),
        (PORTABLE, '9d62d3d21a174e9be7105e4224b4b4b986b89dc6085ec60cc99b3aa2fdd7da77')]:
        closed = read(folder / 'closed.json')
        assert closed['passed'] and pin(folder / 'closed.json')['sha256'] == sha
        for name, wanted in closed['files'].items():
            assert pin(ROOT / name) == wanted, name
        for identity in closed['identities']:
            try:
                assert psutil.Process(identity['pid']).create_time() != identity['birth']
            except psutil.NoSuchProcess:
                pass
    primary = ROOT / 'artifacts/pyannote-amd-execution-v4-20260921'
    assert pin(primary / 'closed.json')['sha256'] == 'e0f1123a85c2a575e062fbf0e4dacc8c742c087661c2599d0197e5ca35cecc68'
    assert read(primary / 'closed.json')['passed']
    verified_files(primary, read(primary / 'closed.json')['files'])
    old = read(OLD / 'payload.json')
    verified_files(OLD, old['files'])
    assert psutil.virtual_memory().available >= 8 * 1024**3
    PREPARED.mkdir()
    BASE.mkdir()
    payload = PREPARED / 'payload'
    payload.mkdir()
    for name in old['files']:
        if name.startswith(('source/', 'runtimes/', 'tools/')) or name == 'prospective-plan.md':
            continue
        copy(OLD / name, payload / name)
    shutil.copytree(COMBINED / 'source', payload / 'source', ignore=shutil.ignore_patterns('bin', 'obj'))
    cores = {}
    for role, directory in [('production', OLD / 'runtimes/rows'), ('portable', PORTABLE / 'runtime'), ('rows', COMBINED / 'runtime')]:
        target = payload / 'runtimes' / role
        shutil.copytree(OLD / 'runtimes/rows', target)
        for p in directory.glob('*.dll'):
            copy(p, target / p.name)
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            copy(PORTABLE / 'runtime' / ('NaturalMeetings.' + suffix), target / ('NaturalMeetings.' + suffix))
        cores[role] = pin(target / 'Lokad.Onnx.dll')
        for family in ['pyannote', 'parakeet']:
            path = payload / 'manifests' / (role + '-' + family + '.json')
            manifest = read(OLD / 'manifests' / ('rows-' + family + '.json'))
            manifest.update(product_source=ROLE_LABELS[role], core_sha256=cores[role]['sha256'],
                data_sha256=pin(target / 'Lokad.Onnx.Data.dll')['sha256'])
            path.unlink()
            write(path, manifest)
    assert {k: v['sha256'] for k, v in cores.items()} == dict(
        production='29477d505dd230aef0b5aa2792da8ec76c6903c5d2cff244327d803b432b4cbb',
        portable='e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838',
        rows='e36963d848ff907acc521908c3a82569dd096c9e0e9ccad864a0d26028082d80')
    meetings = payload / 'meetings'
    meetings.mkdir()
    shutil.copytree(PORTABLE / 'meetings/inputs', meetings / 'inputs')
    manifest = read(PORTABLE / 'meetings/manifest.json')
    manifest.update(core_sha256=cores['rows']['sha256'],
        data_sha256=pin(payload / 'runtimes/rows/Lokad.Onnx.Data.dll')['sha256'],
        accuracy_scope='Actual combined AMD candidate; original two long meetings, recovery and native references.')
    write(meetings / 'manifest.json', manifest)
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
    files = {p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()}
    spec = dict(old)
    spec.update(source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        files=files, cores=cores, external=external, remote=REMOTE, role_labels=ROLE_LABELS,
        source_files={p.relative_to(payload / 'source').as_posix(): pin(p) for p in (payload / 'source').rglob('*') if p.is_file()},
        predecessor=pin(OLD / 'payload.json'), combined_qualification=pin(COMBINED / 'closed.json'),
        portable_qualification=pin(PORTABLE / 'closed.json'),
        scope='Actual combined candidate against current portable-only, previous rows and fresh ORT; includes both natural meetings and recovery. No execution yet.')
    # Legacy role keys are stable protocol identifiers, explicitly mapped above.
    spec['protocol'] = dict(old['protocol'], natural_meetings=3, performance_admission=pin(TOOLS / 'admission.py'))
    for obsolete in ['test_harness_adaptation']:
        spec.pop(obsolete, None)
    write(payload / 'payload.json', spec)
    with tarfile.open(PREPARED / 'payload.tar.gz', 'w:gz') as archive:
        for name in [*sorted(files), 'payload.json']:
            archive.add(payload / name, arcname=name, recursive=False)
    prepared = dict(passed=True, payload=pin(payload / 'payload.json'), archive=pin(PREPARED / 'payload.tar.gz'),
        files=len(files), bytes=sum(v['bytes'] for v in files.values()),
        local_qualification=pin(COMBINED / 'closed.json'), scope=spec['scope'])
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
    prior_execution = read(primary / 'execution/execution.json')
    execution = dict(schema=1, source=spec['source'], payload=prepared['payload'], payload_archive=prepared['archive'],
        remote=REMOTE, limits=LIMITS, local_tools=local_tools, files={p.name: pin(p) for p in target.iterdir() if p.is_file()},
        external=prior_execution['external'], native_predecessor=pin(primary / 'execution/execution.json'),
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
