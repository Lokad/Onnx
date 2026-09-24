"""Use the current application's complete dependencies for unchanged product inspection."""
import json
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.boot_time() == 1789634288.0
    assert psutil.virtual_memory().available >= LIMITS['preflight_available'] and psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for label in ['parent','reference']:
        folder = Path(stage[label]); prefix = 'original' if label == 'parent' else 'reference'
        for name in ['collection.json','payload.json']:
            assert pin(folder/name) == pin(BASE/'evidence'/(prefix+'-'+name))
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code'] == (1 if label == 'parent' else 0) and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        for name, wanted in read(folder/'payload.json')['files'].items(): assert pin(folder/name) == wanted, name
    for name, row in stage['links'].items():
        source = Path(row['source']); target = (BASE/name).resolve()
        assert target.is_relative_to(BASE.resolve()) and not target.exists() and pin(source) == row['identity']
        target.parent.mkdir(parents=True, exist_ok=True); os.link(source,target)
        assert pin(target) == row['identity']
    previous = read(BASE/'evidence/original-payload.json')
    for name, wanted in previous['external'].items(): assert pin(name) == wanted, name
    for folder, names in [('measured', stage['measured']),('runtime',stage['product'])]:
        for name,wanted in names.items(): assert pin(BASE/folder/name) == wanted
    save(BASE/'built.json', dict(passed=True, product=stage['product'],
        files={n:r['identity'] for n,r in stage['links'].items() if n.startswith('runtime/')}))
    (BASE/'source').mkdir()
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, boot_time=1789634288.0,
        previous_owner=read(BASE/'evidence/original-identity.json')['supervisor'], measured=stage['measured'],
        feed=previous['feed'], external=previous['external'], interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        scope='Only repeat failed inspection with complete verified reference dependencies; unchanged binaries, no compilation or inference.')
    save(BASE/'payload.json',payload); verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__ == '__main__': main()
