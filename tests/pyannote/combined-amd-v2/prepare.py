"""Retain successful AMD stages; replace only two corrected graph consumers."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from candidate_protocol import LIMITS, pin, read, write, verified_files
from transport import ROOT, BASE, PREPARED, TOOLS, REMOTE, SITE
from resume_prefix import retained, PREFIX

PRIOR_PAYLOAD = ROOT / 'artifacts/pyannote-combined-amd-payload-20260922/payload'
PRIOR = ROOT / 'artifacts/pyannote-combined-amd-execution-20260922'
CONSUMERS = ROOT / 'artifacts/pyannote-combined-consumers-20260922'


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    assert not PREPARED.exists() and not BASE.exists()
    assert pin(CONSUMERS / 'closed.json')['sha256'] == '602d752956d6cf830851cd0dc35eb4bdf1230880fb2dce6d8bbea6f530bb555c'
    closed = read(CONSUMERS / 'closed.json')
    assert closed['passed']
    for name, wanted in closed['files'].items():
        assert pin(ROOT / name) == wanted
    sys.path.insert(0, str(SITE))
    import psutil
    for identity in closed['identities']:
        try:
            assert psutil.Process(identity['pid']).create_time() != identity['birth']
        except psutil.NoSuchProcess:
            pass
    assert pin(PRIOR / 'failure-closed.json')['sha256'] == 'afdda8b80915238c975c47f18e76e1ce7d99a032beeaa76de1ac5e780bdd601a'
    verified_files(PRIOR, read(PRIOR / 'failure-closed.json')['files'])
    old = read(PRIOR_PAYLOAD / 'payload.json')
    verified_files(PRIOR_PAYLOAD, old['files'])
    assert psutil.virtual_memory().available >= 8 * 1024**3
    PREPARED.mkdir()
    BASE.mkdir()
    payload = PREPARED / 'payload'
    payload.mkdir()
    for name in old['files']:
        copy(PRIOR_PAYLOAD / name, payload / name)
    changes = {}
    for role in ['portable', 'rows']:
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            name = 'runtimes/' + role + '/GraphQualification.' + suffix
            copy(CONSUMERS / role / 'bin/Release/net10.0' / ('GraphQualification.' + suffix), payload / name)
            if pin(payload / name) != old['files'][name]:
                changes[name] = dict(before=old['files'][name], after=pin(payload / name))
    assert all(name.startswith(('runtimes/portable/GraphQualification.', 'runtimes/rows/GraphQualification.')) for name in changes)
    assert {'runtimes/portable/GraphQualification.dll', 'runtimes/rows/GraphQualification.dll'} <= set(changes)
    for name, wanted in old['files'].items():
        assert pin(payload / name) == (changes[name]['after'] if name in changes else wanted), name
    previous = payload / 'predecessor'
    copy(PRIOR / 'failure-closed.json', previous / 'failure-closed.json')
    shutil.copytree(PRIOR / 'collected', previous / 'collected')
    retained(payload)
    built = read(PRIOR / 'collected/campaign/built-files.json')
    for name, wanted in built.items():
        assert pin(PRIOR / 'collected' / name) == wanted
        copy(PRIOR / 'collected' / name, payload / name)
    copy(CONSUMERS / 'closed.json', payload / 'consumer-correction.json')
    spec = dict(old)
    spec.update(remote=REMOTE, files={p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()},
        source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        correction=dict(consumers=changes, consumer_qualification=pin(CONSUMERS / 'closed.json'),
            prior_failure=pin(PRIOR / 'failure-closed.json'), reused_stages=PREFIX, product_files_changed=False,
            timing_started_in_prior=False),
        scope='Two consumer Data-digest literal corrections; reuse fourteen closed successful AMD stages; execute remaining original protocol without changing products or timing gates.')
    # The admission source is byte-identical; only its tool location changes.
    assert pin(TOOLS / 'admission.py') == spec['protocol']['performance_admission']
    write(payload / 'payload.json', spec)
    with tarfile.open(PREPARED / 'payload.tar.gz', 'w:gz') as archive:
        for name in [*sorted(spec['files']), 'payload.json']:
            archive.add(payload / name, arcname=name, recursive=False)
    prepared = dict(passed=True, payload=pin(payload / 'payload.json'), archive=pin(PREPARED / 'payload.tar.gz'),
        files=len(spec['files']), bytes=sum(v['bytes'] for v in spec['files'].values()), correction=spec['correction'])
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
    assert result.returncode == 0, 'Selftests failed; preserve preparation.'
    original_execution = read(PRIOR / 'execution/execution.json')
    execution = dict(schema=1, source=spec['source'], payload=prepared['payload'], payload_archive=prepared['archive'],
        remote=REMOTE, limits=LIMITS, local_tools=local_tools, files={p.name: pin(p) for p in target.iterdir() if p.is_file()},
        external=original_execution['external'], selftest=pin(BASE / 'selftest.log'), scope=spec['scope'])
    write(target / 'execution.json', execution)
    with tarfile.open(BASE / 'execution.tar.gz', 'w:gz') as archive:
        for p in sorted(target.iterdir()):
            archive.add(p, arcname=p.name, recursive=False)
    bundle = dict(passed=True, execution=pin(target / 'execution.json'), archive=pin(BASE / 'execution.tar.gz'),
        payload_archive=prepared['archive'], files=len(execution['files']), selftest=execution['selftest'],
        amd_deployed=False, amd_qualified=False, amd_timing=False)
    write(BASE / 'prepared.json', bundle)
    print(dict(payload=prepared['archive'], execution=bundle['archive'], changed_consumers=changes, reused_stages=len(PREFIX)))


if __name__ == '__main__':
    main()
