"""Preserve the missing-helper selftest failure before any AMD deployment."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/pyannote-single-panel-amd-execution-20260922'
PAYLOAD = ROOT / 'artifacts/pyannote-single-panel-amd-payload-20260922'
TOOLS = ROOT / 'tests/pyannote/single-panel-amd'
sys.path.insert(0, str(ROOT / 'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def main():
    assert not (BASE / 'failure-closed.json').exists()
    log = (BASE / 'selftest.log').read_text()
    assert 'Ran 29 tests' in log and 'FAILED (errors=1)' in log and log.count(' ... ok') == 28
    assert "ModuleNotFoundError: No module named 'prepare_execution'" in log
    assert not (TOOLS / 'prepare_execution.py').exists()
    assert not any((BASE / name).exists() for name in ['prepared.json', 'staged.json', 'deployment.json', 'controller'])
    prepared = read(PAYLOAD / 'prepared.json')
    assert prepared['passed'] and prepared['archive'] == pin(PAYLOAD / 'payload.tar.gz')
    assert prepared['payload'] == pin(PAYLOAD / 'payload/payload.json')
    for name, wanted in read(PAYLOAD / 'payload/payload.json')['files'].items():
        assert pin(PAYLOAD / 'payload' / name) == wanted, name
    own = psutil.Process().pid
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        if p.pid == own or not (p.info['name'] or '').lower().startswith('python'): continue
        args = [a.replace('\\', '/').lower() for a in p.info['cmdline'] or []]
        assert not any(a.endswith('tests/pyannote/single-panel-amd/prepare.py') or a.endswith('tests/pyannote/single-panel-amd') for a in args), p.pid
    files = {p.relative_to(ROOT).as_posix(): pin(p) for folder in [TOOLS, BASE, PAYLOAD] for p in folder.rglob('*') if p.is_file()}
    files[Path(__file__).resolve().relative_to(ROOT).as_posix()] = pin(Path(__file__).resolve())
    result = dict(passed=True, expected_preparation_failure=True, selftests=29, passed_tests=28, errors=1,
                  cause='A required inventory helper was omitted from the copied tools; no numerical assertion failed.',
                  exec_session=8500, observed_exit_code=1, no_live_preparation_command=True,
                  no_deployment=True, no_vm_work=True, files=files)
    with (BASE / 'failure-closed.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(closure=pin(BASE / 'failure-closed.json'), files=len(files), no_deployment=True)))


if __name__ == '__main__': main()
