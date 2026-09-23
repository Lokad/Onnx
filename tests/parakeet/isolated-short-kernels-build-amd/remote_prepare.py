"""Link the selected actual root and reuse its offline build environment."""
import json
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
QUALIFIED = Path('/dev/shm/lokad-pyannote-winograd-product-root-20260923')


def main():
    psutil.Process().cpu_affinity([0])
    idle()
    assert not (BASE / 'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE / 'stage.json')
    for name, wanted in stage['files'].items():
        assert pin(BASE / name) == wanted, name
    for name in ['collection.json', 'payload.json']:
        assert pin(QUALIFIED / name) == pin(BASE / 'evidence' / name)
    receipt = read(QUALIFIED / 'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    assert not any(live(i) for i in receipt['identities'])
    previous = read(QUALIFIED / 'payload.json')
    for name, wanted in previous['files'].items():
        assert pin(QUALIFIED / name) == wanted, name
    for name, wanted in previous['external'].items():
        assert pin(name) == wanted, name
    target = BASE / 'measured'
    target.mkdir()
    for name, wanted in stage['measured_files'].items():
        source = QUALIFIED / 'runtime' / name
        assert pin(source) == wanted
        os.link(source, target / name)
        assert pin(target / name) == wanted
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS,
        previous_owner=receipt['identities'][0], boot_time=1789634288.0,
        measured=stage['measured'], feed=previous['feed'],
        external=previous['external'], interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()},
        scope='M50 normal build: preserve shared kernels/general dispatcher, four caller substitutions, six private/internal additions. No numerical or performance admission.')
    save(BASE / 'payload.json', payload)
    verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE / 'payload.json'), files=len(payload['files']))))


if __name__ == '__main__':
    main()
