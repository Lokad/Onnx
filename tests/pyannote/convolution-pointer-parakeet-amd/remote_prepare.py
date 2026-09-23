"""Reuse existing pinned model/reference assets; never rebuild either product."""
import copy
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
MODELS = Path('/dev/shm/lokad-pyannote-convolution-pointer-models-v2-20260923')
APP = Path('/dev/shm/lokad-pyannote-blocked-spatial-app-20260922')


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for folder, digest in [(MODELS, 'd590b1170643193ded8a9d92624d80a88318a95c3d3e84b3ceea215a38563fcb'),
                           (APP, '229556d67d87085875ade0fc027a1b1df327f5578c5c60d27ad961462aac534f')]:
        assert pin(folder/'payload.json')['sha256'] == digest
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        for identity in receipt['identities']: assert not live(identity)
        for name, wanted in read(folder/'payload.json')['files'].items(): assert pin(folder/name) == wanted, name
    models = read(MODELS/'payload.json'); app = read(APP/'payload.json')
    assert stage['identities'] == models['identities']
    external = dict(models['external'])
    for name, wanted in app['external'].items():
        assert name not in external or external[name] == wanted; external[name] = wanted
    for name, wanted in external.items(): assert pin(name) == wanted, name
    for name in ['assets', 'parakeet-reference']: shutil.copytree(APP/name, BASE/name)
    assert pin(BASE/'parakeet-reference/manifest.json')['sha256'] == '3bad7d262b8809b1265c84c8e66d02ee38e7d4cff2d92014448976a9e161103c'
    (BASE/'manifests').mkdir(); (BASE/'runtimes').mkdir()
    for role in ['selected', 'candidate']:
        folder = BASE/'runtimes'/role; shutil.copytree(APP/'runtimes/portable', folder)
        for name, wanted in stage['identities'][role].items():
            source = MODELS/'runtimes'/role/name
            assert pin(source) == wanted; shutil.copy2(source, folder/name)
        for name, wanted in stage['consumers'].items(): assert pin(folder/(name+'.dll')) == wanted
        manifest = copy.deepcopy(read(BASE/'evidence/original-manifest.json'))
        assert manifest == read(APP/'manifests/portable-parakeet.json')
        manifest.update(core_sha256=stage['identities'][role]['Lokad.Onnx.dll']['sha256'],
                        data_sha256=stage['identities'][role]['Lokad.Onnx.Data.dll']['sha256'])
        manifest['product_source'] = 'selected M22 measured Core208371f6/Datab9358370' if role == 'selected' else 'M28 measured Coree776cec2/Data0c55b650; convolution row pointers and fixed spatial steps'
        save(BASE/'manifests'/(role+'-parakeet.json'), manifest)
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, boot_time=1789634288.0,
        previous_owner=read(MODELS/'collection.json')['identities'][0], identities=stage['identities'], consumers=stage['consumers'],
        external=external, interpreter=models['interpreter'],
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and p.name != 'transfer.tar.gz'},
        scope='784 native arrays and twenty complete public Parakeet clips per role, exact selected outputs; no performance score.')
    save(BASE/'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), files=len(payload['files']))))


if __name__ == '__main__': main()
