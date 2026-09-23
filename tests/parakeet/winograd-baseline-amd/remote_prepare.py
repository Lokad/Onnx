"""Reuse unchanged measured runtimes, assets, manifests and the full native inventory."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live
from checks import prereqs

BASE = Path(__file__).resolve().parents[1]
APP = Path('/dev/shm/lokad-pyannote-winograd-product-app-20260923')
ROOT_BUILD = Path('/dev/shm/lokad-pyannote-winograd-product-root-20260923')


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for folder, digest in [(APP, '60e309165247076e489357616f7a4821ac28478fc7577b1df8ce98fe36cc2951'),
                           (ROOT_BUILD, 'ada75b3d7b4a728609b9a9234dc868215e69899b7752cb3d3edd41b524188ade')]:
        assert pin(folder/'payload.json')['sha256'] == digest
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert all(not live(identity) for identity in receipt['identities'])
        for name, wanted in read(folder/'payload.json')['files'].items(): assert pin(folder/name) == wanted, name
    app = read(APP/'payload.json')
    for name, wanted in app['external'].items(): assert pin(name) == wanted, name
    for name in ['assets', 'runtime']: shutil.copytree(APP/name, BASE/name)
    (BASE/'manifests').mkdir(); (BASE/'runtimes').mkdir()
    shutil.copytree(APP/'runtimes/candidate', BASE/'runtimes/current')
    shutil.copy2(APP/'manifests/candidate-parakeet.json', BASE/'manifests/current-parakeet.json')
    assert pin(BASE/'manifests/current-parakeet.json') == pin(BASE/'evidence/original-parakeet.json')
    for name, wanted in stage['identities']['current'].items(): assert pin(BASE/'runtimes/current'/name) == wanted
    for name, wanted in stage['consumers'].items(): assert pin(BASE/'runtimes/current'/(name+'.dll')) == wanted
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, boot_time=1789634288.0,
        previous_owner=receipt['identities'][0], identities=stage['identities'], consumers=stage['consumers'],
        prerequisites=stage['prerequisites'], external=app['external'], interpreter=app['interpreter'], python_paths=app['python_paths'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name != 'transfer.tar.gz'},
        scope='Current integrated Parakeet versus ORT: four fresh timing processes, twenty clips, 320 requests; no product change.')
    prereqs(BASE, payload); save(BASE/'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), files=len(payload['files']), external=len(payload['external']))))


if __name__ == '__main__': main()
