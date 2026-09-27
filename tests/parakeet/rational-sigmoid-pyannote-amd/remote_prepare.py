"""Hardlink the exact qualified consumer, original assets and measured products."""
import copy
import json
import os
from pathlib import Path
import shutil
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
CURRENT = Path('/dev/shm/lokad-parakeet-observed-dense-where-pyannote-20260924')
PREVIOUS = Path('/dev/shm/lokad-parakeet-pad-current-pyannote-20260926')
PRODUCT = Path('/dev/shm/lokad-parakeet-rational-sigmoid-models-20260927')
APP = Path('/dev/shm/lokad-parakeet-rational-sigmoid-app-20260927')
SHARED = Path('/dev/shm/lokad-parakeet-rational-sigmoid-shared-20260927')


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available'] and psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    external = {}
    for folder, label in [(CURRENT, 'current'), (PREVIOUS, 'previous'), (PRODUCT, 'product'), (APP, 'app'), (SHARED, 'shared')]:
        assert pin(folder/'payload.json') == pin(BASE/'evidence'/(label+'-payload.json'))
        assert pin(folder/'collection.json') == pin(BASE/'evidence'/(label+'-collection.json'))
        receipt = read(folder/'collection.json'); spec = read(folder/'payload.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        for name, wanted in spec['files'].items(): assert pin(folder/name) == wanted, name
        for name, wanted in spec['external'].items():
            assert name not in external or external[name] == wanted
            external[name] = wanted
    for name, wanted in external.items(): assert pin(name) == wanted, name
    assert read(BASE/'evidence/app-closed.json')['admitted']
    assert stage['identities'] == read(BASE/'evidence/shared-analysis.json')['identities'] == read(PRODUCT/'payload.json')['identities']
    reuse = read(BASE/'evidence/consumer-reuse.json')
    assert reuse['passed'] and reuse['identities'] == stage['identities'] and reuse['consumer'] == stage['consumer']
    prior = read(PREVIOUS/'built.json')
    assert pin(PREVIOUS/'built.json') == pin(BASE/'evidence/previous-built.json')
    assert prior['passed'] and prior['consumer'] == stage['consumer']
    previous_files = read(PREVIOUS/'collection.json')['files']
    for path in (PREVIOUS/'runtimes/candidate').iterdir():
        if path.is_file(): assert pin(path) == previous_files[path.relative_to(PREVIOUS).as_posix()]
    for name in ['assets', 'graph-reference']:
        shutil.copytree(CURRENT/name, BASE/name, copy_function=os.link)
    assert pin(BASE/'graph-reference.json') == read(CURRENT/'payload.json')['files']['graph-reference.json']
    (BASE/'manifests').mkdir(); (BASE/'runtimes').mkdir()
    shutil.copytree(BASE/'consumer', BASE/'built', copy_function=os.link)
    for role in ['selected', 'candidate']:
        folder = BASE/'runtimes'/role
        shutil.copytree(PREVIOUS/'runtimes/candidate', folder, copy_function=os.link)
        for name, wanted in stage['identities'][role].items():
            source = PRODUCT/'runtimes'/role/name; assert pin(source) == wanted
            (folder/name).unlink(); os.link(source, folder/name)
        for name in ['GraphQualification.dll', 'GraphQualification.deps.json', 'GraphQualification.runtimeconfig.json']:
            assert pin(folder/name) == pin(BASE/'consumer'/name)
        assert pin(folder/'GraphQualification.dll') == stage['consumer']
        manifest = copy.deepcopy(read(BASE/'evidence/original-manifest.json'))
        assert manifest == read(CURRENT/'evidence/original-manifest.json')
        manifest.update(core_sha256=stage['identities'][role]['Lokad.Onnx.dll']['sha256'], data_sha256=stage['identities'][role]['Lokad.Onnx.Data.dll']['sha256'])
        manifest['product_source'] = 'Qualified current root' if role == 'selected' else 'Rational sigmoid candidate'
        save(BASE/'manifests'/(role+'-pyannote.json'), manifest)
    save(BASE/'built.json', dict(passed=True, consumer=stage['consumer'], reused=True,
        files={p.relative_to(BASE).as_posix(): pin(p) for folder in [BASE/'built', BASE/'runtimes'] for p in folder.rglob('*') if p.is_file()}))
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, boot_time=1789634288.0,
        previous_owner=read(SHARED/'collection.json')['identities'][0], identities=stage['identities'],
        consumer=stage['consumer'], external=external, interpreter=read(PRODUCT/'payload.json')['interpreter'],
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and p.name != 'transfer.tar.gz'},
        scope='Reused identity-parameterized consumer; 18 arrays and 16 complete public calls per product; original native bounds and ownership; bounded cross-product floats; no score.')
    save(BASE/'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), files=len(payload['files']), external=len(external))))


if __name__ == '__main__': main()
