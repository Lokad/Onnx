"""Bind the terminal padding build and immutable offline dependencies."""
import json
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
BUILD = Path('/dev/shm/lokad-parakeet-pad-first-use-build-20260923')


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE / 'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['build_preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE / 'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE / name) == wanted, name
    for name in ['collection.json', 'payload.json']:
        assert pin(BUILD / name) == pin(BASE / 'evidence' / ('build-' + name))
    receipt = read(BUILD / 'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    assert not any(live(i) for i in receipt['identities'])
    previous = read(BUILD / 'payload.json')
    for name, wanted in previous['files'].items(): assert pin(BUILD / name) == wanted, name
    for name, wanted in previous['external'].items(): assert pin(name) == wanted, name
    for name, wanted in stage['products']['candidate'].items():
        assert pin(BUILD / 'runtime' / name) == wanted
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, products=stage['products'],
        previous_owner=receipt['identities'][0], boot_time=1789634288.0,
        feed=previous['feed'], external=previous['external'], interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()},
        scope='Four fresh selected/candidate/candidate/selected processes; twelve synthetic public Pad cases, 600 warmups and 180 measurements per case. No model inference or release admission.')
    save(BASE / 'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE / 'payload.json'), files=len(payload['files']), external=len(payload['external']))))


if __name__ == '__main__':
    main()
