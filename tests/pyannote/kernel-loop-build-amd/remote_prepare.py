"""Pin the current measured product and existing SDK/offline feed on AMD."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
CURRENT = Path('/dev/shm/lokad-parakeet-current-baseline-20260922')
QUALIFIED = Path('/dev/shm/lokad-pyannote-lstm-input-root-20260922')


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available'] and psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for folder, digest in [(CURRENT,'8bdc588c30094ff248e393f38dbd2ab1dc2e1e06d811a415921783233c21b188'),
                           (QUALIFIED,'e4ba54a21dbbe3b1158bc37c1304ee5d906bba9fb9e7bbf73a83ace8aab389b4')]:
        assert pin(folder/'payload.json')['sha256'] == digest
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        for identity in receipt['identities']: assert not live(identity)
        for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    measured = BASE/'measured'; measured.mkdir()
    for name, wanted in stage['measured'].items():
        source = CURRENT/'runtimes/current'/name
        assert pin(source) == wanted; shutil.copy2(source, measured/name)
    previous = read(QUALIFIED/'payload.json'); external = dict(previous['external'])
    for name, wanted in external.items(): assert pin(name) == wanted, name
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, previous_owner=read(CURRENT/'collection.json')['identities'][0],
        boot_time=1789634288.0, measured=stage['measured'], feed=previous['feed'], external=external,
        interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Normal isolated M23 CLI build and exact one-method change inventory; numerical qualification and timing remain separate.')
    save(BASE/'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), files=len(payload['files']))))


if __name__ == '__main__': main()
