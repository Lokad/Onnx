"""Pin inherited application workers, checks and exact-clock scorers to their closed run."""
import hashlib
import json
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'pad-current-pyannote-app-amd'


def pin(path):
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify_scope():
    frozen = json.loads((ROOT/'artifacts/parakeet-pad-current-pyannote-app-amd-20260926/prepared.json').read_text())['files']
    files = {}
    for path in PARENT.iterdir():
        if path.is_file():
            name = path.relative_to(ROOT).as_posix()
            assert pin(path) == frozen[name], name
            files[name] = pin(path)
    expected = (PARENT/'remote_prepare.py').read_text()
    for case in ['pyannote','models','shared','app']:
        before = 'parakeet-pad-current-'+case+'-20260926'
        assert before in expected
        expected = expected.replace(before, 'parakeet-rational-sigmoid-'+case+'-20260927')
    expected = expected.replace('parakeet-pad-current-graphs-v2-20260926', 'parakeet-rational-sigmoid-graphs-20260927')
    expected = expected.replace('Current-root padding dispatcher', 'Fixed rational sigmoid candidate')
    assert (TOOLS/'remote_prepare.py').read_text() == expected
    assert not any((TOOLS/name).exists() for name in ['remote.py','protocol.py','checks.py','admission.py','semantics.py'])
    return files


if __name__ == '__main__':
    print(json.dumps(dict(passed=True, frozen_files=len(verify_scope()))))
