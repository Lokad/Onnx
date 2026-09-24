"""Require terminal current-product owners and reuse exact runtime/SDK/feed inputs."""
import json
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE/'payload.json').exists() and psutil.boot_time() == 1789634288.0
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    parent = Path(stage['parent_remote'])
    for kind in ['build','capture']:
        for name in ['collection','state']:
            assert pin(parent/f'{kind}-{name}.json') == pin(BASE/'evidence'/f'parent-{kind}-{name}.json')
        receipt = read(parent/f'{kind}-collection.json'); state = read(parent/f'{kind}-state.json')
        assert receipt['terminal'] and receipt['code'] == 0 and state['complete'] and state['code'] == 0
        owners = [state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
        assert not any(live(i) for i in owners)
    assert pin(parent/'spec.json') == pin(BASE/'evidence/parent-spec.json')
    old = read(parent/'spec.json')
    for name, wanted in old['files'].items(): assert pin(parent/name) == wanted, name
    for name, wanted in old['external'].items(): assert pin(name) == wanted, name
    target = BASE/'measured'; target.mkdir()
    for name, wanted in stage['measured_files'].items():
        source = parent/'source/runtime-observed'/name
        assert pin(source) == wanted
        os.link(source, target/name); assert pin(target/name) == wanted
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, previous_owner=state['supervisor'],
        boot_time=1789634288.0, measured=stage['measured'], feed=stage['feed'], external=stage['external'],
        interpreter=stage['interpreter'], parent_release=stage['parent_release'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        scope='Observed Parakeet layouts select the exact retained dense scalar Where helper on the qualified release; build scope only.')
    save(BASE/'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), files=len(payload['files']))))


if __name__ == '__main__': main()
