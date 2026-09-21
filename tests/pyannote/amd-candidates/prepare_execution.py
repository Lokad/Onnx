"""Freeze the tested execution tools separately from the immutable model payload."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from candidate_protocol import LIMITS, pin, read, write
from transport import ROOT, BASE, PREPARED, TOOLS, REMOTE


def main():
    verified = read(PREPARED/'preparation-verified.json')
    assert verified['passed'] is True and verified['prepared'] == pin(PREPARED/'prepared.json')
    prepared = read(PREPARED/'prepared.json')
    assert verified['archive'] == prepared['archive'] == pin(PREPARED/'payload.tar.gz')
    BASE.mkdir(); target = BASE/'execution'; target.mkdir()
    local_tools = {}
    for path in sorted(TOOLS.glob('*.py')):
        ast.parse(path.read_text(encoding='utf8'), filename=str(path))
        local_tools[path.relative_to(ROOT).as_posix()] = pin(path)
        shutil.copy2(path, target/path.name)
    original = ROOT/'tests/parakeet/transcribe/audit.py'
    local_tools[original.relative_to(ROOT).as_posix()] = pin(original)
    shutil.copy2(original, target/'parakeet_audit.py')
    with (BASE/'selftest.log').open('x') as log:
        code = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover', '-s', str(TOOLS), '-p', 'test_*.py', '-v'],
                              cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120).returncode
    assert code == 0, 'Execution selftests failed; retain this attempt'
    old = ROOT/'artifacts/audio-amd-two-family-20260920/collected/frozen.json'
    closed = read(ROOT/'artifacts/audio-amd-two-family-20260920/closed.json')
    key = next(k for k in closed['files'] if Path(k).resolve() == old.resolve())
    assert pin(old) == closed['files'][key]
    retained = read(old); external = {}
    packages = ('numpy', 'scipy', 'torch', 'torchaudio', 'onnxruntime', 'pyannote', 'einops', 'pandas', 'sortedcontainers', 'psutil')
    for name, wanted in retained['external'].items():
        segment = name.split('/site-packages/', 1)[-1] if '/site-packages/' in name else name.split('/python/', 1)[-1] if '/python/' in name else ''
        package = segment.split('/')[0]
        if any(package == p or package.startswith(p+'.') or package.startswith(p+'-') or package.startswith(p+'_') for p in packages):
            external[name] = wanted
    assert external and not any('/transformers/' in name for name in external)
    execution = dict(schema=1, source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                     payload=prepared['payload'], payload_archive=prepared['archive'], remote=REMOTE, limits=LIMITS,
                     local_tools=local_tools, files={p.name: pin(p) for p in target.iterdir() if p.is_file()}, external=external,
                     native_predecessor=pin(old), selftest=pin(BASE/'selftest.log'),
                     scope='Tested offline execution tools; AMD inference has not run. Existing e5 owns the VM until terminal.')
    write(target/'execution.json', execution)
    with tarfile.open(BASE/'execution.tar.gz', 'w:gz') as archive:
        for path in sorted(target.iterdir()): archive.add(path, arcname=path.name, recursive=False)
    receipt = dict(passed=True, execution=pin(target/'execution.json'), archive=pin(BASE/'execution.tar.gz'),
                   payload_archive=prepared['archive'], local_preparation=pin(PREPARED/'preparation-verified.json'),
                   files=len(execution['files']), supplemental_native_files=len(external),
                   supplemental_native_bytes=sum(v['bytes'] for v in external.values()), selftest=execution['selftest'],
                   amd_deployed=False, amd_qualified=False, amd_timing=False)
    write(BASE/'prepared.json', receipt); print(json.dumps(receipt))


if __name__ == '__main__':
    main()
