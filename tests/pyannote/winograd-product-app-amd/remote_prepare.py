"""Reuse closed products, complete native inventory and all meeting fixtures."""
import copy
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
from checks import prereqs

BASE = Path(__file__).resolve().parents[1]
APP = Path('/dev/shm/lokad-pyannote-blocked-spatial-app-20260922')
MODELS = Path('/dev/shm/lokad-pyannote-winograd-product-models-v2-20260923')
PREVIOUS = [
    ('lokad-pyannote-winograd-product-20260923', '3ee8122e3f0894608eea68d22c73c01b64de48406ab42704670964cd685eafc7'),
    ('lokad-pyannote-winograd-product-models-v2-20260923', '7aada241e75543efa75fb492c1090b813fa6fe8d25b12c7c3b2f9ac39461c47c'),
    ('lokad-pyannote-winograd-product-parakeet-20260923', 'f53070c9efd1b3457790076f05ad90f849e0296c2a229b0e9843daf8094d7b5b'),
    ('lokad-pyannote-winograd-product-shared-v2-20260923', '900c4c67385f7f1f87620ac3b9e487f01907ac0757e90a06e3419622310b9761'),
]


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name,wanted in stage['files'].items(): assert pin(BASE/name) == wanted,name
    for suffix,digest in PREVIOUS:
        folder = Path('/dev/shm')/suffix
        assert pin(folder/'payload.json')['sha256'] == digest
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        for identity in receipt['identities']: assert not live(identity)
        for name,wanted in read(folder/'payload.json')['files'].items(): assert pin(folder/name) == wanted,name
    assert pin(APP/'payload.json')['sha256'] == '229556d67d87085875ade0fc027a1b1df327f5578c5c60d27ad961462aac534f'
    app = read(APP/'payload.json')
    for name,wanted in app['files'].items(): assert pin(APP/name) == wanted,name
    assert pin(APP/'execution/execution.json') == stage['native_inventory'] == pin(BASE/'evidence/original-execution.json')
    external = dict(read(MODELS/'payload.json')['external'])
    for source in [app['external'],read(BASE/'evidence/original-execution.json')['external']]:
        for name,wanted in source.items():
            assert name not in external or external[name] == wanted; external[name] = wanted
    for name,wanted in external.items(): assert pin(name) == wanted,name
    for name in ['assets','runtime','meetings']: shutil.copytree(APP/name,BASE/name)
    (BASE/'manifests').mkdir(); (BASE/'runtimes').mkdir()
    for role in ['selected','candidate']:
        folder = BASE/'runtimes'/role; shutil.copytree(APP/'runtimes/portable',folder)
        for name,wanted in stage['identities'][role].items():
            source = MODELS/'runtimes'/role/name; assert pin(source) == wanted; shutil.copy2(source,folder/name)
        for name,wanted in stage['consumers'].items(): assert pin(folder/(name+'.dll')) == wanted
        for family in ['pyannote','parakeet']:
            manifest = copy.deepcopy(read(BASE/'evidence'/('original-'+family+'.json')))
            assert manifest == read(APP/'manifests'/('portable-'+family+'.json'))
            manifest.update(core_sha256=stage['identities'][role]['Lokad.Onnx.dll']['sha256'],data_sha256=stage['identities'][role]['Lokad.Onnx.Data.dll']['sha256'])
            manifest['product_source'] = 'selected M22 measured Core208371f6/Datab9358370' if role == 'selected' else 'M34 measured Core521bae17/Dataf3b9aa81; admitted Winograd convolution'
            save(BASE/'manifests'/(role+'-'+family+'.json'),manifest)
    meeting = read(BASE/'meetings/manifest.json'); assert meeting == read(BASE/'evidence/original-meetings.json')
    meeting.update(core_sha256=stage['identities']['candidate']['Lokad.Onnx.dll']['sha256'],data_sha256=stage['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256'])
    save(BASE/'meetings/manifest.json',meeting)
    payload = dict(passed=True,jobs=JOBS,limits=LIMITS,boot_time=1789634288.0,prerequisites=stage['prerequisites'],
        previous_owner=receipt['identities'][0],identities=stage['identities'],consumers=stage['consumers'],
        external=external,interpreter=app['interpreter'],python_paths=app['python_paths'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name != 'transfer.tar.gz'},
        scope='Fresh native Pyannote/Parakeet public checks, both600smeetings and30srecovery, fixed96request Pyannote matched comparison.')
    prereqs(BASE,payload); save(BASE/'payload.json',payload); verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(external))))


if __name__ == '__main__': main()
