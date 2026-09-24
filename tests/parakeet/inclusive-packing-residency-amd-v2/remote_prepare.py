"""Bind qualified packing products and existing actual model files."""
import json
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
BUILD = Path('/dev/shm/lokad-parakeet-inclusive-packing-build-20260924')
CONTRACTS = Path('/dev/shm/lokad-parakeet-inclusive-packing-contracts-v2-20260924')


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE / 'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE / 'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE / name) == wanted, name
    for name in ['collection.json', 'payload.json']: assert pin(BUILD / name) == pin(BASE / 'evidence' / name)
    receipt = read(BUILD / 'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    assert not any(live(i) for i in receipt['identities'])
    assert pin(CONTRACTS / 'collection.json') == pin(BASE / 'evidence/contracts/collection.json')
    contracts = read(CONTRACTS / 'collection.json')
    assert contracts['terminal'] and contracts['code'] == 0 and contracts['input_error'] is None
    assert not any(live(i) for i in contracts['identities'])
    assert pin(CONTRACTS / 'payload.json') == pin(BASE / 'evidence/contracts/payload.json')
    for name, wanted in read(CONTRACTS / 'payload.json')['files'].items(): assert pin(CONTRACTS / name) == wanted, name
    previous = read(BUILD / 'payload.json')
    for name, wanted in previous['files'].items(): assert pin(BUILD / name) == wanted, name
    for name, wanted in previous['external'].items(): assert pin(Path(name)) == wanted, name
    for name, wanted in stage['identities']['candidate'].items(): assert pin(BUILD / 'runtime' / name) == wanted, name
    for name, wanted in stage['model_files'].items(): assert pin(Path(name)) == wanted, name
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, identities=stage['identities'], models=stage['models'], budgets=stage['budgets'],
                   previous_owner=receipt['identities'][0], boot_time=1789634288.0,
                   feed=previous['feed'], external=previous['external'] | stage['model_files'], interpreter=previous['interpreter'],
                   parent_build=stage['parent_build'], files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()},
                   scope='Actual encoder/decoder residency at unchanged 256/64 MiB budgets, selected and candidate, both modes. All names/content/ownership/invalidation audited; no inference, product rebuild or performance claim.')
    save(BASE / 'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE / 'payload.json'), files=len(payload['files']))))


if __name__ == '__main__': main()
