"""Independently close focused frontend qualification after actual termination."""
import common
from common import *
from phase_audit import audit_preparation

if __name__ == '__main__':
    assert not (BASE / 'focused-closed.json').exists()
    analysis = audit_preparation(BASE, common)
    assert read(BASE / 'focused.json')['sparse_cases'] == 58
    assert read(BASE / 'hardware-disabled.json')['sparse_cases'] == 58
    instructions = read(BASE / 'instructions.json')
    assert instructions['passed']
    save(BASE / 'focused-analysis.json', analysis)
    files = dict(read(BASE / 'prepared.json')['files'])
    for path in [*BASE.rglob('*'), *TOOLS.iterdir(), ROOT / 'tests/pyannote/convolution-pool/phase_audit.py']:
        if path.is_file() and not {'obj', 'packages'}.intersection(path.parts):
            files[rel(path)] = pin(path)
    save(BASE / 'focused-closed.json', dict(passed=True, files=files, analysis=pin(BASE / 'focused-analysis.json'), identities=analysis['identities']))
    print(json.dumps(dict(closed=pin(BASE / 'focused-closed.json'), resources=analysis['resource_samples'], identities=len(analysis['identities']))))
