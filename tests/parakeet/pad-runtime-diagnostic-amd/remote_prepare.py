"""Link immutable inputs only after all preceding diagnostic owners terminate."""
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
idle(); assert psutil.boot_time() == 1789634288.0 and not (BASE / 'payload.json').exists()
stage = read(BASE / 'stage.json')
for label, remote in [('screen', '/dev/shm/lokad-parakeet-pad-dispatch-screen-20260923'),
                      ('events', '/dev/shm/lokad-graph-startup-diagnostic-20260923')]:
    receipt = read(BASE / 'evidence' / (label + '-collection.json'))
    assert receipt['terminal'] and receipt['code'] == 0
    assert not any(live(i) for i in receipt['identities'])
    assert pin(Path(remote) / 'collection.json') == pin(BASE / 'evidence' / (label + '-collection.json'))
for name, wanted in stage['files'].items(): assert pin(BASE / name) == wanted, name
for name, link in stage['links'].items():
    target = (BASE / name).resolve()
    assert target.is_relative_to(BASE.resolve()) and not target.exists()
    source = Path(link['source']); assert pin(source) == link['identity'], str(source)
    target.parent.mkdir(parents=True, exist_ok=True); os.link(source, target)
payload = dict(passed=True, diagnostic_only=True, jobs=JOBS, limits=LIMITS,
    products=stage['products'], exporter=stage['exporter'], boot_time=psutil.boot_time(),
    previous_owner=read(BASE / 'evidence/screen-collection.json')['identities'][0],
    feed=stage['feed'], external=stage['external'], interpreter=stage['interpreter'],
    files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()})
save(BASE / 'payload.json', payload); verify(BASE)
print('Unchanged products and complete event exporter prepared')
