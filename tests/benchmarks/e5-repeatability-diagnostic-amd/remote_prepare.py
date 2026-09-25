"""Link immutable inputs only after the Parakeet application is closed."""
import json
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]


def main():
    idle()
    assert psutil.boot_time() == 1789634288.0 and not (BASE / 'payload.json').exists()
    # Leave room for the entire existing stage allowance before allocating traces.
    assert psutil.virtual_memory().available >= LIMITS['preflight_available'] + LIMITS['artifacts']
    stage = read(BASE / 'stage.json')
    for name, wanted in stage['files'].items():
        assert pin(BASE / name) == wanted, name
    for label, remote in stage['receipts'].items():
        path = BASE / 'evidence' / (label + '-collection.json')
        receipt = read(path)
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        assert pin(Path(remote)) == pin(path), label
    for name, link in stage['links'].items():
        target = (BASE / name).resolve()
        assert target.is_relative_to(BASE.resolve()) and not target.exists()
        source = Path(link['source'])
        assert pin(source) == link['identity'], str(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    application = read(BASE / 'evidence/application-collection.json')
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, products=stage['products'],
        previous_owner=application['identities'][0], previous_consumer=stage['previous_consumer'],
        boot_time=psutil.boot_time(), feed=stage['feed'], external=stage['external'], interpreter=stage['interpreter'],
        roundtrip=stage['roundtrip'], storage_estimate=stage['storage_estimate'],
        failed_release_controls=stage['failed_release_controls'], diagnostic_only=True, release_admitted=False,
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE / 'payload.json', payload)
    verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE / 'payload.json'))))


if __name__ == '__main__':
    main()
