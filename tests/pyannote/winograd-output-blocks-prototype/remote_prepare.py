"""Pin actual normal product binaries and existing native fixtures before workers."""
import json
from pathlib import Path
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE = Path(__file__).resolve().parents[1]
BUILD = Path('/dev/shm/lokad-pyannote-winograd-output-blocks-build-20260923')
FIXTURES = Path('/dev/shm/lokad-pyannote-blocked-spatial-product-20260922/fixtures')


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['build_preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name,wanted in stage['files'].items(): assert pin(BASE/name) == wanted,name
    assert pin(BUILD/'collection.json') == pin(BASE/'evidence/build-collection.json')
    receipt = read(BUILD/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    assert all(not live(identity) for identity in receipt['identities'])
    assert pin(BUILD/'payload.json') == pin(BASE/'evidence/build-payload.json')
    for name,wanted in read(BUILD/'payload.json')['files'].items(): assert pin(BUILD/name) == wanted,name
    previous = read(BUILD/'payload.json'); external = dict(previous['external'])
    for name,wanted in stage['fixtures'].items():
        p = FIXTURES/name; assert pin(p) == wanted,name; external[str(p)] = wanted
    for name,wanted in external.items(): assert pin(name) == wanted,name
    for role,files in stage['products'].items():
        for name,wanted in files.items(): assert pin(BASE/'runtimes'/role/name) == wanted
    payload = dict(passed=True,jobs=JOBS,limits=LIMITS,products=stage['products'],previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        feed=previous['feed'],external=external,interpreter=previous['interpreter'],fixtures=str(FIXTURES),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Actual current/candidate Core DLLs, same numerical driver, complete original and expanded cases; no timing.')
    save(BASE/'payload.json',payload); verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(external))))


if __name__ == '__main__': main()
