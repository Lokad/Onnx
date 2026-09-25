"""Verify existing VM model assets and install unchanged Replay with both cores."""
import json
import os
from pathlib import Path
import shutil
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
RELEASE = Path('/dev/shm/lokad-parakeet-slice-dense-conversion-models-20260925')
MODELS = Path('/dev/shm/lokad-parakeet-owned-batch-isolation-models-20260925')
APP = Path('/dev/shm/lokad-parakeet-owned-batch-isolation-release-app-20260925')


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    proof=read(BASE/'evidence/models-closed.json'); assert proof['passed']
    assert proof['analysis']==pin(BASE/'evidence/models-analysis.json')
    assert read(BASE/'evidence/app-closed.json')['admitted']
    for folder, label in [(RELEASE, 'release'), (MODELS, 'models'), (APP, 'app')]:
        assert pin(folder/'collection.json') == pin(BASE/'evidence'/(label+'-collection.json'))
        assert pin(folder/'payload.json') == pin(BASE/'evidence'/(label+'-payload.json'))
        closed = read(folder/'collection.json')
        assert closed['terminal'] and closed['code'] == 0 and closed['input_error'] is None
        assert not any(live(i) for i in closed['identities'])
        for name, wanted in read(folder/'payload.json')['files'].items(): assert pin(folder/name) == wanted, name
    models = read(MODELS/'payload.json'); receipt = read(MODELS/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    for identity in receipt['identities']: assert not live(identity)
    for name, wanted in models['files'].items(): assert pin(MODELS/name) == wanted, name
    release=read(RELEASE/'payload.json')
    assert stage['identities']==dict(selected=release['identities']['selected'],candidate=models['identities']['candidate'])
    assert read(BASE/'evidence/app-analysis.json')['identities'] == dict(current=stage['identities']['selected'], candidate=stage['identities']['candidate'])
    external = dict(models['external'])
    for name,wanted in release['external'].items():
        assert name not in external or external[name]==wanted; external[name]=wanted
    for name, wanted in stage['model_assets'].items():
        assert name not in external or external[name] == wanted; external[name] = wanted
    for name, wanted in external.items(): assert pin(name) == wanted, name
    (BASE/'runtimes').mkdir()
    for role in ['selected','candidate']:
        folder = BASE/'runtimes'/role; shutil.copytree(BASE/'consumer', folder, copy_function=os.link)
        source = (RELEASE if role=='selected' else MODELS)/'runtimes'/role/'Lokad.Onnx.dll'
        assert pin(source) == stage['identities'][role]['Lokad.Onnx.dll']; os.link(source, folder/source.name)
        assert not (folder/'Lokad.Onnx.Data.dll').exists()
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, boot_time=1789634288.0,
        previous_owner=read(APP/'collection.json')['identities'][0], identities=stage['identities'], consumer=stage['consumer'], runtime=stage['runtime'],
        model_assets=stage['model_assets'], external=external, interpreter=models['interpreter'],
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and p.name != 'transfer.tar.gz'},
        scope='Original complete shared-model and five-input e5 qualification, exact selected/native outputs; no performance score.')
    save(BASE/'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), files=len(payload['files']))))


if __name__ == '__main__': main()
