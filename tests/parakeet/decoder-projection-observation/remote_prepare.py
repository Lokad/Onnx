"""Bind terminal qualification owners and hardlink the existing immutable tools."""
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
PRIOR = dict(root=Path('/dev/shm/lokad-parakeet-rational-sigmoid-root-20260927'),
             events=Path('/dev/shm/lokad-parakeet-dispatch-events-20260923'),
             export=Path('/dev/shm/lokad-parakeet-dispatch-full-export-20260923'))


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE/'payload.json').exists()
    assert psutil.boot_time() == 1789634288.0
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for label, folder in PRIOR.items():
        for name in ['payload.json', 'collection.json']:
            assert pin(folder/name) == pin(BASE/'evidence'/label/name)
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
    for name, link in stage['links'].items():
        target = (BASE/name).resolve()
        assert target.is_relative_to(BASE.resolve()) and not target.exists()
        source = Path(link['source'])
        assert pin(source) == link['identity']
        target.parent.mkdir(parents=True, exist_ok=True); os.link(source, target)
    prior = read(PRIOR['root']/'payload.json')
    external = dict(prior['external'])
    external[stage['model_path']] = stage['model']
    for name, wanted in external.items(): assert pin(name) == wanted, name
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, product=stage['product'],
        previous_owner=read(PRIOR['root']/'collection.json')['identities'][0], boot_time=1789634288.0,
        feed=prior['feed'], external=external, interpreter=prior['interpreter'],
        root_qualification=stage['root_qualification'],
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and p.name != 'transfer.tar.gz'},
        scope='One original decoder fixture: ordinary control and sampled execution; no performance admission.')
    save(BASE/'payload.json', payload); verify(BASE)
    print(dict(passed=True, files=len(payload['files']), external=len(external)))


if __name__ == '__main__': main()
