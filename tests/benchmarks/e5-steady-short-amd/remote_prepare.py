"""Link the exact terminal inputs and reserve new benchmark consumer outputs."""
import copy
import json
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]


def main():
    psutil.Process().cpu_affinity([0])
    idle()
    assert psutil.boot_time() == 1789634288.0 and not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items():
        assert pin(BASE/name) == wanted, name
    for label, remote in stage['receipts'].items():
        local = BASE/'evidence'/label/'collection.json'
        assert pin(remote) == pin(local), label
        receipt = read(local)
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert not any(live(identity) for identity in receipt['identities'])
    for name, wanted in stage['external'].items():
        assert pin(name) == wanted, name
    for name, link in stage['links'].items():
        target = (BASE/name).resolve()
        assert target.is_relative_to(BASE.resolve()) and not target.exists()
        source = Path(link['source'])
        assert pin(source) == link['identity'], str(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    assert pin(BASE/'previous/ReleaseBenchmark.dll') == stage['previous_consumer']
    for role in ['current', 'candidate']:
        assert pin(BASE/'runtimes'/role/'Lokad.Onnx.dll') == stage['products'][role]['Lokad.Onnx.dll']
        cases = copy.deepcopy(read(BASE/'cases.json'))
        cases['core'] = stage['products'][role]['Lokad.Onnx.dll']['sha256']
        save(BASE/f'cases-{role}.json', cases)
    fields = ['previous_owner', 'feed', 'external', 'interpreter', 'python_paths', 'products', 'previous_consumer']
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, boot_time=1789634288.0,
        **{name: stage[name] for name in fields},
        files={p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE/'payload.json', payload)
    verify(BASE)
    print(json.dumps(dict(passed=True, files=len(payload['files']), external=len(payload['external']))))


if __name__ == '__main__':
    main()
