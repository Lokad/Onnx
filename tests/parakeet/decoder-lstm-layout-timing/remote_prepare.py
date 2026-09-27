"""Bind original terminal products, diagnosed contracts, fixtures and offline feed."""
import os
from pathlib import Path
import sys
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
PARENTS = dict(candidate=Path('/dev/shm/lokad-lstmlayout3-20260927'),
    control=Path('/dev/shm/lokad-lstmlayout-baseline-20260927'),
    root=Path('/dev/shm/lokad-parakeet-decoder-packed-row-root-20260927'),
    calls=Path('/dev/shm/lokad-parakeet-prepared-recurrence-calls-v2-20260924'))


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE/'payload.json').exists() and psutil.boot_time() == 1789634288.0
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for label, folder in PARENTS.items():
        assert pin(folder/'collection.json') == pin(BASE/'evidence'/label/'collection.json')
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['input_error'] is None
        assert receipt['code'] == (1 if label == 'candidate' else 0)
        assert not any(live(owner) for owner in receipt['identities'])
    for name, link in stage['links'].items():
        source = (PARENTS['calls']/link['source']).resolve(); target = (BASE/name).resolve()
        assert source.is_relative_to(PARENTS['calls']) and target.is_relative_to(BASE)
        assert pin(source) == link['pin']; target.parent.mkdir(parents=True, exist_ok=True); os.link(source, target)
    prior = read(PARENTS['root']/'payload.json'); external = dict(prior['external'])
    for name, wanted in external.items(): assert pin(name) == wanted, name
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, identities=stage['identities'],
        previous_owner=read(PARENTS['control']/'collection.json')['identities'][0],
        boot_time=1789634288.0, feed=prior['feed'], interpreter=prior['interpreter'],
        python_paths=['/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'],
        external=external, files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE/'payload.json', payload); verify(BASE)
    print(dict(passed=True, files=len(payload['files']), external=len(external)))


if __name__ == '__main__': main()
