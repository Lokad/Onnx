"""Freeze the qualified normal product for a fresh complete AMD comparison."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from candidate_protocol import LIMITS, pin, read, write, verified_files
from transport import ROOT, BASE, PREPARED, TOOLS, REMOTE, SITE, local_e5_terminal
from fresh_qualification import IDENTITIES, retained_callers
from admission import ROLE_LABELS

OLD = ROOT/'artifacts/pyannote-single-panel-amd-payload-v2-20260922/payload'
MODEL = ROOT/'artifacts/pyannote-blocked-spatial-composition-v3-20260922'
GRAPH = ROOT/'artifacts/pyannote-blocked-spatial-models-20260922'
AMD = ROOT/'artifacts/pyannote-blocked-spatial-product-amd-v3-20260922'
PARAKEET = ROOT/'artifacts/pyannote-blocked-spatial-parakeet-20260922'


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)


def main():
    assert not PREPARED.exists() and not BASE.exists(); local_e5_terminal()
    sys.path.insert(0, str(SITE)); import psutil
    assert psutil.virtual_memory().available >= 8*1024**3
    evidence = {}
    for folder, sha, relative, identities in [
        (MODEL, 'e7a9a30d88ef2a425c0c51b007e2b7d89428d54e50ef7dc4e446e191f95719e3', False, 'identities'),
        (ROOT/'artifacts/pyannote-blocked-spatial-package-20260922', 'fc4f4811032ff38ea837b8d53ea68325a63cd9959b7e5d61b9fbb89ba7fefc66', False, 'identities'),
        (GRAPH, '2cb2905c448861e7d4554a41c436e7eeccfea73a9f5517ea7370d9bb04fcf005', True, 'terminal_identities'),
        (ROOT/'artifacts/pyannote-blocked-spatial-shared-20260922', 'e27a6e668548db02b77f0b0bd6cbd8f6bdced3f260faa6ad39be294e3886fe31', True, 'terminal_identities'),
        (AMD, '88ec9b71d6e7807440ccd65d6148ead8c585a0633ce755898e174b8501b77197', False, 'local_identities'),
        (PARAKEET, '33777770849d4d941904272bc6ff95c51b52253f3b27aab126a585977119d46e', True, 'identities')]:
        path = folder/'closed.json'
        if sha is not None: assert pin(path)['sha256'] == sha
        proof = read(path); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin((ROOT if relative else folder)/name) == wanted, name
        for identity in proof[identities]:
            try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
            except psutil.NoSuchProcess: pass
        evidence[path.relative_to(ROOT).as_posix()] = pin(path)
    assert read(PARAKEET/'analysis.json')['known_native_failures_preserved']
    old = read(OLD/'payload.json'); verified_files(OLD, old['files'])
    PREPARED.mkdir(); BASE.mkdir(); payload = PREPARED/'payload'; payload.mkdir()
    for name in old['files']:
        if name.startswith(('source/', 'runtimes/', 'graph-consumer/', 'manifests/', 'retained/', 'caller/', 'caller-source/')) or name == 'prospective-plan.md': continue
        copy(OLD/name, payload/name)
    shutil.copytree(MODEL/'source', payload/'source', ignore=shutil.ignore_patterns('bin', 'obj'))
    (payload/'manifests').mkdir()
    for role in IDENTITIES:
        source = OLD/'runtimes/portable' if role == 'production' else GRAPH/'runtime'
        shutil.copytree(source, payload/'runtimes'/role)
        for runner in ['AudioBenchmark', 'NaturalMeetings', 'TranscribeReplay']:
            for suffix in ['dll', 'deps.json', 'runtimeconfig.json']: copy(OLD/'runtimes/portable'/(runner+'.'+suffix), payload/'runtimes'/role/(runner+'.'+suffix))
        for name, sha in zip(['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'], IDENTITIES[role], strict=True): assert pin(payload/'runtimes'/role/name)['sha256'] == sha
        for family in ['pyannote', 'parakeet']:
            manifest = read(OLD/'manifests'/('portable-'+family+'.json'))
            manifest.update(core_sha256=IDENTITIES[role][0], data_sha256=IDENTITIES[role][1])
            write(payload/'manifests'/(role+'-'+family+'.json'), manifest)
    for name in ['closed.json', 'analysis.json']: copy(AMD/name, payload/'retained-product'/name)
    retained_callers(payload)
    shutil.copytree(GRAPH/'consumer', payload/'graph-consumer', ignore=shutil.ignore_patterns('bin', 'obj'))
    meeting = payload/'meetings/manifest.json'; manifest = read(meeting)
    manifest.update(core_sha256=IDENTITIES['portable'][0], data_sha256=IDENTITIES['portable'][1])
    meeting.write_text(__import__('json').dumps(manifest, indent=2)+'\n', encoding='utf8')
    copy(TOOLS/'README.md', payload/'prospective-plan.md')
    files = {p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()}
    spec = dict(old)
    spec.update(source='a0cc7741 plus qualified normal prepared-convolution implementation and corrected focused tests', files=files, remote=REMOTE,
        role_labels=ROLE_LABELS, cores={r:pin(payload/'runtimes'/r/'Lokad.Onnx.dll') for r in IDENTITIES},
        source_files={p.relative_to(payload/'source').as_posix():pin(p) for p in (payload/'source').rglob('*') if p.is_file()},
        windows_qualification=evidence, scope='Fresh AMD complete models, long meetings, normal Linux suites and matched selected/candidate/ORT application comparison.')
    spec.pop('retained_qualification', None); spec.pop('graph_consumer_adaptation', None)
    spec['protocol'] = dict(old['protocol'], qualification_reused=False, actual_product_callers_reused=pin(AMD/'closed.json'), performance_admission=pin(TOOLS/'admission.py'))
    spec['protocol'].pop('caller_cases_per_mode', None)
    write(payload/'payload.json', spec)
    with tarfile.open(PREPARED/'payload.tar.gz', 'w:gz') as archive:
        for name in [*sorted(files), 'payload.json']: archive.add(payload/name, arcname=name, recursive=False)
    prepared = dict(passed=True, payload=pin(payload/'payload.json'), archive=pin(PREPARED/'payload.tar.gz'), files=len(files), bytes=sum(v['bytes'] for v in files.values()), scope=spec['scope'])
    write(PREPARED/'prepared.json', prepared)
    target = BASE/'execution'; target.mkdir(); local_tools = dict(evidence)
    for p in sorted(TOOLS.iterdir()):
        if not p.is_file(): continue
        if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), filename=str(p))
        local_tools[p.relative_to(ROOT).as_posix()] = pin(p); copy(p, target/p.name)
    original = ROOT/'tests/parakeet/transcribe/audit.py'
    local_tools[original.relative_to(ROOT).as_posix()] = pin(original); copy(original, target/'parakeet_audit.py')
    env = dict(os.environ, PYTHONPATH=str(SITE), PYTHONDONTWRITEBYTECODE='1'); env.pop('PYTHONOPTIMIZE', None)
    with (BASE/'selftest.log').open('x') as log:
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover', '-s', str(TOOLS), '-p', 'test_*.py', '-v'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0, 'Selftests failed; preserve preparation.'
    previous = read(ROOT/'artifacts/pyannote-single-panel-amd-execution-v2-20260922/execution/execution.json')
    execution = dict(schema=1, source=spec['source'], payload=prepared['payload'], payload_archive=prepared['archive'], remote=REMOTE, limits=LIMITS,
        local_tools=local_tools, files={p.name:pin(p) for p in target.iterdir() if p.is_file()}, external=previous['external'], selftest=pin(BASE/'selftest.log'), scope=spec['scope'])
    write(target/'execution.json', execution)
    with tarfile.open(BASE/'execution.tar.gz', 'w:gz') as archive:
        for p in sorted(target.iterdir()): archive.add(p, arcname=p.name, recursive=False)
    bundle = dict(passed=True, execution=pin(target/'execution.json'), archive=pin(BASE/'execution.tar.gz'), payload_archive=prepared['archive'], files=len(execution['files']), selftest=execution['selftest'], amd_deployed=False, amd_qualified=False, amd_timing=False)
    write(BASE/'prepared.json', bundle); print(dict(payload=prepared, execution=bundle), flush=True)


if __name__ == '__main__': main()
