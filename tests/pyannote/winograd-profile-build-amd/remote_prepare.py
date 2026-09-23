"""Verify closed prerequisites and reuse the pinned SDK/offline feed."""
import json
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
APP = Path('/dev/shm/lokad-pyannote-winograd-product-app-20260923')
ROOT_BUILD = Path('/dev/shm/lokad-pyannote-winograd-product-root-20260923')


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name,wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for folder,label,digest in [
        (APP,'current','60e309165247076e489357616f7a4821ac28478fc7577b1df8ce98fe36cc2951'),
        (ROOT_BUILD,'root','ada75b3d7b4a728609b9a9234dc868215e69899b7752cb3d3edd41b524188ade')]:
        proof = read(BASE/'evidence'/(label+'-closed.json'))
        assert pin(folder/'collection.json') == proof['files']['collected/collection.json']
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert all(not live(identity) for identity in receipt['identities'])
        assert pin(folder/'payload.json')['sha256'] == digest
        for name,wanted in read(folder/'payload.json')['files'].items(): assert pin(folder/name) == wanted,name
    for name,wanted in stage['product'].items(): assert pin(BASE/'runtime'/name) == wanted
    assert pin(BASE/'previous/SampledAudio.dll') == stage['previous_consumer']
    environment = read(ROOT_BUILD/'payload.json')
    for name,wanted in environment['external'].items(): assert pin(name) == wanted,name
    payload = dict(passed=True,jobs=JOBS,limits=LIMITS,product=stage['product'],previous_consumer=stage['previous_consumer'],
        previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        feed=environment['feed'],external=environment['external'],interpreter=environment['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Current Winograd product copied unchanged; only two identity operands in SampledAudio may change.')
    save(BASE/'payload.json',payload); verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__ == '__main__': main()
