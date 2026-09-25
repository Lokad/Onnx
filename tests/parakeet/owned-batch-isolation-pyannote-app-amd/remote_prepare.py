"""Reuse immutable Pyannote application assets with the exact qualified products."""
import copy
import json
import os
from pathlib import Path
import shutil
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live
from checks import prereqs

BASE = Path(__file__).resolve().parents[1]
OLD = Path('/dev/shm/lokad-parakeet-observed-dense-where-pyannote-app-20260924')
MODELS = Path('/dev/shm/lokad-parakeet-owned-batch-isolation-pyannote-20260925')
PRIOR = dict(baseline=OLD, models=MODELS,
    parakeet=Path('/dev/shm/lokad-parakeet-owned-batch-isolation-models-20260925'),
    shared=Path('/dev/shm/lokad-parakeet-owned-batch-isolation-shared-20260925'))
PRIOR['parakeet-app'] = Path('/dev/shm/lokad-parakeet-owned-batch-isolation-release-app-20260925')
PRIOR['parakeet-release'] = Path('/dev/shm/lokad-parakeet-slice-dense-conversion-models-20260925')


def main():
    psutil.Process().cpu_affinity([0])
    idle()
    assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items():
        assert pin(BASE/name) == wanted, name
    for label, folder in PRIOR.items():
        for name in ['payload.json', 'collection.json']:
            assert pin(folder/name) == pin(BASE/'evidence'/label/name)
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert not any(live(identity) for identity in receipt['identities'])
        for name, wanted in read(folder/'payload.json')['files'].items():
            assert pin(folder/name) == wanted, name
    old = read(OLD/'payload.json')
    external = old['external']
    for name, wanted in external.items():
        assert pin(name) == wanted, name
    for name in ['assets', 'runtime', 'meetings']:
        shutil.copytree(OLD/name, BASE/name, copy_function=os.link)
    (BASE/'manifests').mkdir()
    (BASE/'runtimes').mkdir()
    for role in ['selected', 'candidate']:
        folder = BASE/'runtimes'/role
        shutil.copytree(OLD/'runtimes/candidate', folder, copy_function=os.link)
        for name, wanted in stage['identities'][role].items():
            source = MODELS/'runtimes'/role/name
            assert pin(source) == wanted
            (folder/name).unlink()
            os.link(source, folder/name)
        for name, wanted in stage['consumers'].items():
            assert pin(folder/(name+'.dll')) == wanted
        for family in ['pyannote', 'parakeet']:
            manifest = copy.deepcopy(read(BASE/'evidence'/('original-'+family+'.json')))
            assert manifest == read(OLD/'manifests'/('candidate-'+family+'.json'))
            manifest.update(core_sha256=stage['identities'][role]['Lokad.Onnx.dll']['sha256'],
                            data_sha256=stage['identities'][role]['Lokad.Onnx.Data.dll']['sha256'])
            manifest['product_source'] = ('Qualified release Coref95a13c5/Dataa893952f' if role == 'selected'
                                          else 'Dispatch relocation Coree07a4518/Data01e9e784')
            save(BASE/'manifests'/(role+'-'+family+'.json'), manifest)
    meeting = read(BASE/'meetings/manifest.json')
    assert meeting == read(BASE/'evidence/original-meetings.json')
    meeting.update(core_sha256=stage['identities']['candidate']['Lokad.Onnx.dll']['sha256'],
                   data_sha256=stage['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256'])
    save(BASE/'meetings/manifest.json', meeting)
    assert pin(BASE/'evidence/selected-meetings.json') == pin(OLD/'meetings-run/output/result.json')
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, boot_time=1789634288.0,
        prerequisites=stage['prerequisites'], graph_qualification=stage['graph_qualification'],
        previous_owner=read(MODELS/'collection.json')['identities'][0],
        identities=stage['identities'], consumers=stage['consumers'], external=external,
        interpreter=old['interpreter'], python_paths=old['python_paths'],
        files={path.relative_to(BASE).as_posix(): pin(path) for path in BASE.rglob('*') if path.is_file()},
        scope='Original nine jobs: fresh native Pyannote, both600smeetings and30srecovery,96 timed requests,complete release outputs exact,all original regression limits.')
    prereqs(BASE, payload)
    save(BASE/'payload.json', payload)
    verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), files=len(payload['files']), external=len(external))))


if __name__ == '__main__':
    main()
