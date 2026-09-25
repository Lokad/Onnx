"""Link the exact released and relocation products after parent comparison closes."""
import json
import os
import shutil
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live
from checks import prereqs

BASE = Path(__file__).resolve().parents[1]
CURRENT = Path('/dev/shm/lokad-parakeet-winograd-baseline-20260923')
MODELS = Path('/dev/shm/lokad-parakeet-owned-batch-isolation-models-20260925')
RELEASE_MODELS = Path('/dev/shm/lokad-parakeet-slice-dense-conversion-models-20260925')
PARENT_APP = Path('/dev/shm/lokad-parakeet-owned-batch-isolation-parent-app-20260925')


def main():
    psutil.Process().cpu_affinity([0])
    idle()
    assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items():
        assert pin(BASE/name) == wanted, name
    for label, folder in [('baseline', CURRENT), ('models', MODELS), ('release_models', RELEASE_MODELS), ('parent_app', PARENT_APP)]:
        local = BASE/'evidence'/label
        assert pin(folder/'collection.json') == pin(local/'collection.json')
        receipt = read(local/'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert not any(live(identity) for identity in receipt['identities'])
        if label != 'parent_app':
            assert pin(folder/'payload.json') == pin(local/'payload.json')
            for name, wanted in read(folder/'payload.json')['files'].items():
                assert pin(folder/name) == wanted, name
    original = read(CURRENT/'payload.json')
    for name, wanted in original['external'].items():
        assert pin(name) == wanted, name
    for name in ['assets', 'runtime']:
        shutil.copytree(CURRENT/name, BASE/name, copy_function=os.link)
    (BASE/'manifests').mkdir()
    (BASE/'runtimes').mkdir()
    for role, folder, source in [('current', RELEASE_MODELS, 'selected'), ('candidate', MODELS, 'candidate')]:
        shutil.copytree(folder/'runtimes'/source, BASE/'runtimes'/role, copy_function=os.link)
        shutil.copy2(BASE/'evidence'/(role+'-parakeet.json'), BASE/'manifests'/(role+'-parakeet.json'))
        for name, wanted in stage['identities'][role].items():
            assert pin(BASE/'runtimes'/role/name) == wanted
        for name, wanted in stage['consumers'].items():
            assert pin(BASE/'runtimes'/role/(name+'.dll')) == wanted
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, boot_time=1789634288.0,
        previous_owner=read(PARENT_APP/'collection.json')['identities'][0],
        identities=stage['identities'], consumers=stage['consumers'], prerequisites=stage['prerequisites'],
        failed_graph_cases=[], release_admitted=False,
        external=original['external'], interpreter=original['interpreter'], python_paths=original['python_paths'],
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()},
        scope='Actual release f95,relocation e07,ORT,ORT,relocation e07,release f95;20 clips;480 requests;original full-call scoring and numerical policy. Root admission remains separate.')
    prereqs(BASE, payload)
    save(BASE/'payload.json', payload)
    verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), files=len(payload['files']), external=len(payload['external']))))


if __name__ == '__main__':
    main()
