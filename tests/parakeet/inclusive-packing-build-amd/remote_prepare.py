"""Require the terminal parent release and reuse immutable binaries and offline inputs."""
import json
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
QUALIFIED = Path('/dev/shm/lokad-parakeet-wide-entry-first-use-build-20260923')
RELEASE = Path('/dev/shm/lokad-parakeet-wide-entry-first-use-root-v2-20260923')


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE / 'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE / 'stage.json')
    for name, wanted in stage['files'].items():
        assert pin(BASE / name) == wanted, name
    for folder, evidence in [(QUALIFIED, BASE / 'evidence'), (RELEASE, BASE / 'evidence/release')]:
        for name in ['collection.json', 'payload.json']:
            assert pin(folder / name) == pin(evidence / name)
        receipt = read(folder / 'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        for name, wanted in read(folder / 'payload.json')['files'].items():
            assert pin(folder / name) == wanted, name
    previous = read(QUALIFIED / 'payload.json')
    for name, wanted in previous['external'].items():
        assert pin(name) == wanted, name
    target = BASE / 'measured'; target.mkdir()
    for name, wanted in stage['measured_files'].items():
        source = QUALIFIED / 'runtime' / name
        assert pin(source) == wanted
        os.link(source, target / name)
        assert pin(target / name) == wanted
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS,
                   previous_owner=receipt['identities'][0], boot_time=1789634288.0,
                   measured=stage['measured'], feed=previous['feed'], external=previous['external'],
                   interpreter=previous['interpreter'], parent_release=stage['parent_release'],
                   files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()},
                   scope='M63 normal build: only the GraphPacking.FitsPackBudget inclusive 4096 comparison. All budgets and kernels unchanged; no numerical or performance admission.')
    save(BASE / 'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE / 'payload.json'), files=len(payload['files']))))


if __name__ == '__main__':
    main()
