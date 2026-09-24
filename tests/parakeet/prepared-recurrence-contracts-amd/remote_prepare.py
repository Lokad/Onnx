"""Keep the qualified products and offline inputs exact for focused tests."""
import json
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
BUILD = Path('/dev/shm/lokad-parakeet-prepared-recurrence-build-20260924')


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE / 'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE / 'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE / name) == wanted, name
    for name in ['collection.json', 'payload.json']: assert pin(BUILD / name) == pin(BASE / 'evidence' / name)
    receipt = read(BUILD / 'collection.json')
    assert receipt['terminal'] and receipt['code'] == 1 and receipt['input_error'] is None
    assert not any(live(i) for i in receipt['identities'])
    review=read(BASE/'evidence/review-closed.json');assert review['passed']
    assert pin(BASE/'evidence/review-analysis.json')==review['analysis']
    accepted=read(BASE/'evidence/review-analysis.json')
    assert accepted['original_refusal']==pin(BASE/'evidence/closed.json')
    assert accepted['raw_inventory']==pin(BUILD/'inventory/instructions.json')
    assert accepted['built']==stage['identities']['candidate'] and accepted['rebuilds']==0
    previous = read(BUILD / 'payload.json')
    for name, wanted in previous['files'].items(): assert pin(BUILD / name) == wanted, name
    for name, wanted in previous['external'].items(): assert pin(Path(name)) == wanted, name
    for name, wanted in stage['identities']['candidate'].items(): assert pin(BUILD / 'runtime' / name) == wanted, name
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, identities=stage['identities'], expected_cases=stage['expected_cases'],
                   previous_owner=receipt['identities'][0], boot_time=1789634288.0,
                   feed=previous['feed'], external=previous['external'], interpreter=previous['interpreter'],
                   parent_build=stage['parent_build'], files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()},
                   scope='Focused recurrence, existing LSTM and aggregate packing contracts only. Selected public budget failure required; both candidate instruction modes must pass the full frozen census. No product rebuild or performance claim.')
    save(BASE / 'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE / 'payload.json'), files=len(payload['files']))))


if __name__ == '__main__': main()
