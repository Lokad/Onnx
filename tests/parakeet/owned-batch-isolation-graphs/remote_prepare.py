"""Hardlink the unchanged comparison inputs and replace only candidate Core."""
from pathlib import Path
import copy
import json
import os
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from checks import consumer_inventory, e5_consumer_inventory
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE/'payload.json').exists()
    stage = read(BASE/'stage.json')
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for remote, local in [('/dev/shm/lokad-parakeet-packed-final-row-graphs-20260925/collection.json', 'evidence/parent-graphs/collection.json'),
                          ('/dev/shm/lokad-parakeet-owned-batch-isolation-build-20260925/capture-collection.json', 'evidence/isolation-build/collection.json')]:
        assert pin(remote) == pin(BASE/local)
        receipt = read(remote)
        assert receipt['terminal'] and receipt['code'] == 0 and not any(live(i) for i in receipt['identities'])
    for name, wanted in stage['external'].items(): assert pin(name) == wanted, name
    for name, value in stage['links'].items():
        destination = BASE/name
        assert destination.resolve().is_relative_to(BASE.resolve()) and not destination.exists()
        source = Path(value['source']); assert pin(source) == value['identity'], name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
    for role in ['current', 'candidate']:
        cases = copy.deepcopy(read(BASE/'cases.json'))
        cases['core'] = stage['products'][role]['Lokad.Onnx.dll']['sha256']
        save(BASE/('cases-'+role+'.json'), cases)
        for prefix in ['runtimes','runtimes-e5']:
            assert pin(BASE/prefix/role/'Lokad.Onnx.dll') == stage['products'][role]['Lokad.Onnx.dll']
    built = read(BASE/'built.json')
    for name, wanted in built['files'].items(): assert pin(BASE/name) == wanted, name
    review = consumer_inventory(read(BASE/'evidence/warmed-consumer/instructions.json'), stage, built)
    assert review == read(BASE/'evidence/warmed-consumer/review.json')
    e5 = e5_consumer_inventory(read(BASE/'evidence/e5-consumer/instructions.json'),
                             dict(previous_consumer=stage['consumer']), dict(consumer=built['e5_consumer']))
    assert e5 == read(BASE/'evidence/e5-consumer/review.json')
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, previous_owner=stage['previous_owner'], boot_time=1789634288.0,
                   **{name:stage[name] for name in ['external','interpreter','python_paths','products','previous_consumer','consumer','e5_consumer']},
                   files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name != 'transfer.tar.gz'})
    save(BASE/'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), files=len(payload['files']))))


if __name__ == '__main__':
    main()
