"""Apply the five qualified source changes only after full application admission."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prepared_patch', HERE/'prepare_patch_v3.py')
patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
ROOT = patch.ROOT
APP = ROOT/'artifacts/pyannote-lstm-input-app-amd-20260922'
BASE = ROOT/'artifacts/pyannote-lstm-input-root-integration-20260922'


def main():
    assert not BASE.exists()
    assert patch.pin(APP/'closed.json')['sha256'] == '73a4897a4db8e1bd729cb3c5486bcb11814d4ff669c9b9a472003572b08c64d0'
    closed = patch.read(APP/'closed.json'); assert closed['passed']
    for name, wanted in closed['files'].items(): assert patch.pin(APP/name) == wanted, name
    analysis = patch.read(APP/'analysis.json')
    sys.path.insert(0, str(ROOT/'tests/pyannote/lstm-input-app-amd'))
    from admission import evaluate
    assert analysis['passed'] and evaluate(analysis['table']) == analysis['performance']
    assert analysis['performance']['admitted']
    assert (analysis['native_public_requests'], analysis['meeting_requests'], analysis['timing_requests']) == (24, 3, 96)
    assert patch.pin(patch.BASE/'prepared.json')['sha256'] == '70653927725ee3137ac09755584b6313c145ab6d04496c4ebbbb2c5fefc9ac3c'
    prepared = patch.read(patch.BASE/'prepared.json')
    assert prepared['passed'] and prepared['names'] == patch.NAMES
    assert patch.pin(patch.BASE/'candidate.patch') == prepared['patch']
    for name, wanted in prepared['before'].items():
        assert (patch.pin(ROOT/name) if (ROOT/name).exists() else None) == wanted, name
    subprocess.run(['git', 'apply', '--check', '--ignore-space-change', str(patch.BASE/'candidate.patch')], cwd=ROOT, check=True)
    BASE.mkdir()
    subprocess.run(['git', 'apply', '--ignore-space-change', str(patch.BASE/'candidate.patch')], cwd=ROOT, check=True)
    for name in prepared['before']:
        assert patch.canonical(ROOT/name) == patch.canonical(ROOT/prepared['source']/name), name
    receipt = dict(passed=True, application=patch.pin(APP/'closed.json'), prepared=patch.pin(patch.BASE/'prepared.json'),
        source_files={name:patch.pin(ROOT/name) for name in prepared['before']},
        changed=patch.NAMES, root_build_pending=True, tool=patch.pin(Path(__file__)))
    (BASE/'applied.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(passed=True, receipt=patch.pin(BASE/'applied.json'), changed=patch.NAMES)))


if __name__ == '__main__': main()
