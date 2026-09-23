"""Verify terminal predecessors and reuse the qualified SDK/feed inventory."""
import json
import shutil
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
CURRENT = Path('/dev/shm/lokad-parakeet-winograd-baseline-20260923')
LATEST = Path('/dev/shm/lokad-parakeet-short-wide-pack-screen-20260923')
QUALIFIED = Path('/dev/shm/lokad-pyannote-winograd-product-root-20260923')


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name,wanted in stage['files'].items(): assert pin(BASE/name) == wanted,name
    for folder,label in [(CURRENT,'current'),(QUALIFIED,'qualified')]:
        assert pin(folder/'collection.json') == pin(BASE/'evidence'/(label+'-collection.json'))
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert all(not live(identity) for identity in receipt['identities'])
        assert pin(folder/'payload.json') == pin(BASE/'evidence'/(label+'-payload.json'))
        for name,wanted in read(folder/'payload.json')['files'].items(): assert pin(folder/name) == wanted,name
    assert pin(LATEST/'collection.json')==pin(BASE/'evidence/latest-collection.json')
    latest=read(LATEST/'collection.json');assert latest['terminal'] and latest['code']==0 and latest['input_error'] is None
    assert all(not live(i) for i in latest['identities'])
    environment = read(QUALIFIED/'payload.json')
    for name,wanted in environment['external'].items(): assert pin(name) == wanted,name
    measured = BASE/'measured'; measured.mkdir()
    for name,wanted in stage['measured_files'].items():
        source = CURRENT/'runtimes/current'/name; assert pin(source) == wanted,name
        shutil.copy2(source,measured/name)
    for name,wanted in stage['measured'].items(): assert pin(measured/name) == wanted
    payload = dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=latest['identities'][0],boot_time=1789634288.0,
        measured=stage['measured'],feed=environment['feed'],external=environment['external'],interpreter=environment['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Isolated M40 product build and complete one-method inventory; no inference, timing or root integration.')
    save(BASE/'payload.json',payload); verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__ == '__main__': main()
