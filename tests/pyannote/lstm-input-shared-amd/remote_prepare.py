"""Verify existing VM model assets and install unchanged Replay with both cores."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
MODELS = Path('/dev/shm/lokad-pyannote-lstm-input-models-20260922')


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    assert pin(MODELS/'payload.json')['sha256'] == 'c77de9ddf395571e6205837b5cd9fc5691cccff1e9b3da8f7ec85879cba38c64'
    models = read(MODELS/'payload.json'); receipt = read(MODELS/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    for identity in receipt['identities']: assert not live(identity)
    for name, wanted in models['files'].items(): assert pin(MODELS/name) == wanted, name
    assert models['identities'] == stage['identities']
    external = dict(models['external'])
    for name, wanted in stage['model_assets'].items():
        assert name not in external or external[name] == wanted; external[name] = wanted
    for name, wanted in external.items(): assert pin(name) == wanted, name
    (BASE/'runtimes').mkdir()
    for role in ['selected','candidate']:
        folder = BASE/'runtimes'/role; shutil.copytree(BASE/'consumer', folder)
        source = MODELS/'runtimes'/role/'Lokad.Onnx.dll'
        assert pin(source) == stage['identities'][role]['Lokad.Onnx.dll']; shutil.copy2(source, folder/source.name)
        assert not (folder/'Lokad.Onnx.Data.dll').exists()
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, boot_time=1789634288.0,
        previous_owner=receipt['identities'][0], identities=stage['identities'], consumer=stage['consumer'], runtime=stage['runtime'],
        model_assets=stage['model_assets'], external=external, interpreter=models['interpreter'],
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and p.name != 'transfer.tar.gz'},
        scope='Original complete shared-model and five-input e5 qualification, exact selected/native outputs; no performance score.')
    save(BASE/'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), files=len(payload['files']))))


if __name__ == '__main__': main()
