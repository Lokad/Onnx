"""Link verified retained inputs without rebuilding product or observer."""
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
idle()
assert psutil.boot_time() == 1789634288.0 and not (BASE/'payload.json').exists()
stage = read(BASE/'stage.json')
receipt = read(BASE/'evidence/events-collection.json')
assert receipt['terminal'] and receipt['code'] == 0 and not any(live(i) for i in receipt['identities'])
for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
for name, link in stage['links'].items():
    target = (BASE/name).resolve()
    assert target.is_relative_to(BASE.resolve()) and not target.exists()
    source = Path(link['source']); assert pin(source) == link['identity'], str(source)
    target.parent.mkdir(parents=True, exist_ok=True); os.link(source, target)
payload = dict(passed=True, diagnostic_only=True, jobs=JOBS, limits=LIMITS, boot_time=psutil.boot_time(),
    previous_owner=receipt['identities'][0],
    **{k:stage[k] for k in ['products','observer','original_consumer','candidate_methods','exporter','feed','external','interpreter']},
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()})
save(BASE/'payload.json', payload)
verify(BASE)
print('Retained products, observer, models and event tools verified')
