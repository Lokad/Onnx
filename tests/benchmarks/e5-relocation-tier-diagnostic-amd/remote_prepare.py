"""Link the proved observer and exact products after both source campaigns close."""
import json
import os
from pathlib import Path
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live
from checks import compiled_scope, observer_scope
from source_scope import verify as verify_source

BASE = Path(__file__).resolve().parents[1]


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert psutil.boot_time() == 1789634288.0 and not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']+LIMITS['artifacts']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    stage = read(BASE/'stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for label, remote in stage['receipts'].items():
        local = BASE/'evidence'/(label+'-collection.json')
        assert pin(remote) == pin(local), label
        receipt = read(local)
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
        assert not any(live(identity) for identity in receipt['identities'])
    assert pin(BASE/'evidence/parent-closed.json') == stage['parent_closure']
    assert pin(BASE/'evidence/graph-closed.json') == stage['graph_closure']
    graph = read(BASE/'evidence/graph-closed.json'); assert graph['passed'] and not graph['admitted']
    assert read(BASE/'evidence/graph-payload.json')['products'] == stage['products']
    for name, wanted in stage['external'].items(): assert pin(name) == wanted, name
    for name, link in stage['links'].items():
        target = (BASE/name).resolve()
        assert target.is_relative_to(BASE.resolve()) and not target.exists()
        source = Path(link['source']); assert pin(source) == link['identity'], name
        target.parent.mkdir(parents=True, exist_ok=True); os.link(source, target)
    built = read(BASE/'built.json'); assert built['passed'] and built['consumer'] == stage['reused_consumer']
    for name, wanted in built['files'].items(): assert pin(BASE/name) == wanted, name
    for role, product in stage['products'].items():
        assert read(BASE/f'cases-{role}.json')['core'] == product['Lokad.Onnx.dll']['sha256']
        assert pin(BASE/'runtimes'/role/'Lokad.Onnx.dll') == product['Lokad.Onnx.dll']
        assert pin(BASE/'runtimes'/role/'ReleaseBenchmark.dll') == stage['reused_consumer']
    for name, checker in [('consumer-inventory',compiled_scope), ('observer-inventory',observer_scope)]:
        assert checker(read(BASE/name/'instructions.json'), stage, built) == read(BASE/name/'review.json')
    assert verify_source((BASE/'evidence/OriginalProgram.cs.txt').read_text(),
                         (BASE/'source/consumer/Program.cs').read_text()) == read(BASE/'evidence/source-review.json')
    assert built['exporter'] == stage['reused_exporter']['binary'] == pin(BASE/'export-runtime/DispatchEventsExport.dll')
    assert stage['reused_exporter']['roundtrip']['passed']
    fields = ['products','previous_owner','previous_consumer','previous_observer','reused_consumer',
              'reused_exporter','parent_closure','graph_closure','feed','interpreter','external',
              'failed_release_cases','diagnostic_only','release_admitted','storage_estimate']
    payload = dict(passed=True, jobs=JOBS, limits=LIMITS, boot_time=psutil.boot_time(),
                   **{key:stage[key] for key in fields},
                   files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE/'payload.json', payload); verify(BASE)
    print(json.dumps(dict(passed=True, payload=pin(BASE/'payload.json'), observer=built['consumer'], jobs=len(JOBS))))


if __name__ == '__main__':
    main()
