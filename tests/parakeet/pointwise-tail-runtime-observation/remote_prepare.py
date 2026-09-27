"""Bind terminal owners and retain only the unchanged products and trace tools."""
import os
from pathlib import Path
import shutil
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE/'payload.json').exists() and psutil.boot_time() == 1789634288.0
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for terminal in stage['terminals']:
        assert pin(terminal['remote']) == pin(BASE/terminal['local'])
        receipt = read(terminal['remote'])
        assert receipt['terminal'] and receipt['code'] == 0 and receipt.get('input_error') is None
        assert not any(live(i) for i in receipt['identities'])
    for name, link in stage['links'].items():
        target = (BASE/name).resolve(); source = Path(link['source']).resolve()
        assert target.is_relative_to(BASE) and not target.exists() and pin(source) == link['identity'], name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_dev == target.parent.stat().st_dev: os.link(source, target)
        else: shutil.copy2(source, target)
        assert pin(target) == link['identity']
    for name, wanted in stage['external'].items(): assert pin(name) == wanted, name
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, product=stage['products']['candidate'],
        products=stage['products'], output_hashes=stage['output_hashes'],
        previous_owner=stage['previous_owner'], boot_time=1789634288.0,
        feed=stage['feed'], interpreter=stage['interpreter'], external=stage['external'],
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE/'payload.json', payload); verify(BASE)
    print(dict(passed=True, files=len(payload['files']), external=len(payload['external'])))


if __name__ == '__main__': main()
