"""Bind terminal owners and link retained baseline, captures and observation tools."""
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
PRIOR = Path('/dev/shm/lokad-lstmlayout-timing-20260927')


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE/'payload.json').exists() and psutil.boot_time() == 1789634288.0
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for terminal in stage['terminals']:
        assert pin(terminal['remote']) == pin(BASE/terminal['local'])
        receipt = read(terminal['remote'])
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
    for name, link in stage['links'].items():
        target = (BASE/name).resolve(); source = Path(link['source'])
        assert target.is_relative_to(BASE) and not target.exists() and pin(source) == link['identity'], name
        target.parent.mkdir(parents=True, exist_ok=True); os.link(source, target)
    prior = read(PRIOR/'payload.json'); external = dict(prior['external'])
    for name, wanted in external.items(): assert pin(name) == wanted, name
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, product=stage['product'],
        previous_owner=read(PRIOR/'collection.json')['identities'][0], boot_time=1789634288.0,
        feed=prior['feed'], interpreter=prior['interpreter'], external=external,
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE/'payload.json', payload); verify(BASE)
    print(dict(passed=True, files=len(payload['files']), external=len(external)))


if __name__ == '__main__': main()
