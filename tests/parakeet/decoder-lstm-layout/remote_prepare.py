"""Bind terminal root/capture owners, retained fixtures and the offline feed."""
import os
from pathlib import Path
import sys
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]
PRIOR = Path('/dev/shm/lokad-parakeet-decoder-packed-row-root-20260927')
CAPTURE = Path('/dev/shm/lokad-parakeet-prepared-recurrence-calls-v2-20260924')


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert psutil.boot_time() == 1789634288.0 and not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for label, folder in [('root', PRIOR), ('capture', CAPTURE)]:
        assert pin(folder/'collection.json') == pin(BASE/'evidence'/label/'collection.json')
        receipt = read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert not any(live(owner) for owner in receipt['identities'])
    for name, item in stage['links'].items():
        target = (BASE/name).resolve(); source = Path(item['source'])
        assert target.is_relative_to(BASE.resolve()) and not target.exists()
        assert pin(source) == item['identity']; target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    prior = read(PRIOR/'payload.json'); external = dict(prior['external'])
    for name, wanted in external.items(): assert pin(name) == wanted, name
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, current_product=stage['current_product'],
        previous_owner=read(PRIOR/'collection.json')['identities'][0], boot_time=1789634288.0,
        expected_census=stage['expected_census'], added_tests=stage['added_tests'],
        feed=prior['feed'], external=external, interpreter=prior['interpreter'], python=sys.executable,
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and p.name != 'transfer.tar.gz'})
    save(BASE/'payload.json', payload); verify(BASE)
    print(dict(passed=True, files=len(payload['files']), external=len(external)))


if __name__ == '__main__': main()
