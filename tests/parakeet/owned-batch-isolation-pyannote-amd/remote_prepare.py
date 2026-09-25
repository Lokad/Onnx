"""Link exact released and candidate products with already verified consumers."""
import copy
import json
import os
from pathlib import Path
import shutil
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live
from reuse_checks import verify_reuse

BASE = Path(__file__).resolve().parents[1]
PRODUCT = Path('/dev/shm/lokad-parakeet-owned-batch-isolation-models-20260925')
APP = Path('/dev/shm/lokad-parakeet-owned-batch-isolation-release-app-20260925')
SHARED = Path('/dev/shm/lokad-parakeet-owned-batch-isolation-shared-20260925')
CURRENT = Path('/dev/shm/lokad-parakeet-observed-dense-where-pyannote-20260924')


def main():
    psutil.Process().cpu_affinity([0])
    idle()
    assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items():
        assert pin(BASE/name) == wanted, name
    for folder, label in [(CURRENT, 'current'), (PRODUCT, 'product'), (APP, 'app'), (SHARED, 'shared')]:
        assert pin(folder/'payload.json') == pin(BASE/'evidence'/(label+'-payload.json'))
        assert pin(folder/'collection.json') == pin(BASE/'evidence'/(label+'-collection.json'))
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert not any(live(identity) for identity in receipt['identities'])
        for name, wanted in read(folder/'payload.json')['files'].items():
            assert pin(folder/name) == wanted, name
    current = read(CURRENT/'payload.json')
    current_built = read(CURRENT/'built.json')
    assert pin(CURRENT/'built.json') == read(CURRENT/'collection.json')['files']['built.json']
    for name, wanted in current_built['files'].items():
        assert pin(CURRENT/name) == wanted, name
    assert current['identities']['candidate'] == stage['selected_product']
    external = dict(read(PRODUCT/'payload.json')['external'])
    for name, wanted in current['external'].items():
        assert name not in external or external[name] == wanted
        external[name] = wanted
    for name, wanted in external.items():
        assert pin(name) == wanted, name
    for name in ['assets', 'graph-reference']:
        shutil.copytree(CURRENT/name, BASE/name, copy_function=os.link)
    assert pin(BASE/'graph-reference.json') == current['files']['graph-reference.json']
    (BASE/'manifests').mkdir()
    (BASE/'runtimes').mkdir()
    identities = {}
    for role in ['selected', 'candidate']:
        folder = BASE/'runtimes'/role
        shutil.copytree(CURRENT/'runtimes/candidate', folder, copy_function=os.link)
        if role == 'candidate':
            for name, wanted in stage['product'].items():
                source = PRODUCT/'runtimes/candidate'/name
                assert pin(source) == wanted
                (folder/name).unlink()
                os.link(source, folder/name)
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            name = 'GraphQualification.'+suffix
            (folder/name).unlink()
            os.link(BASE/'consumers'/role/name, folder/name)
        identities[role] = {name: pin(folder/name) for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']}
        manifest = copy.deepcopy(read(BASE/'evidence/original-manifest.json'))
        assert manifest == read(CURRENT/'evidence/original-manifest.json')
        manifest.update(core_sha256=identities[role]['Lokad.Onnx.dll']['sha256'],
                        data_sha256=identities[role]['Lokad.Onnx.Data.dll']['sha256'])
        manifest['product_source'] = ('Qualified release Coref95a13c5/Dataa893952f' if role == 'selected'
                                      else 'Dispatch relocation Coree07a4518/Data01e9e784')
        save(BASE/'manifests'/(role+'-pyannote.json'), manifest)
    assert identities == dict(selected=stage['selected_product'], candidate=stage['product'])
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS,
        previous_owner=read(SHARED/'collection.json')['identities'][0], boot_time=1789634288.0,
        identities=identities, consumers=stage['consumers'], old_data=stage['old_data'], new_data=stage['new_data'],
        external=external, interpreter=read(PRODUCT/'payload.json')['interpreter'],
        files={path.relative_to(BASE).as_posix(): pin(path) for path in BASE.rglob('*') if path.is_file()},
        scope='Two fresh products;18 complete Pyannote arrays and16 public calls each;exact release outputs,original native bounds;reused consumers,no build,no score.')
    verify_reuse(BASE, payload)
    save(BASE/'payload.json', payload)
    verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), files=len(payload['files']))))


if __name__ == '__main__':
    main()
