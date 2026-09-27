"""Reuse qualified full-model inputs and consumers with exact product identities."""
from pathlib import Path
import copy
import json
import os
import psutil
from protocol import JOBS, LIMITS, pin, read, save, verify
from remote import idle, live

BASE = Path(__file__).resolve().parents[1]


def main():
    psutil.Process().cpu_affinity([0]); idle()
    assert not (BASE/'payload.json').exists()
    stage = read(BASE/'stage.json')
    assert psutil.virtual_memory().available >= LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free >= LIMITS['preflight_tmpfs']
    for name, wanted in stage['files'].items(): assert pin(BASE/name) == wanted, name
    for terminal in stage['terminals']:
        assert pin(terminal['remote']) == pin(BASE/terminal['local'])
        receipt = read(terminal['remote'])
        assert receipt['terminal'] and receipt['code'] == 0 and not any(live(i) for i in receipt['identities'])
    compatible = read(BASE/'evidence/compatibility.json')
    assert compatible['passed'] and compatible['original_public_bindings_preserved']
    assert compatible['all_data_methods_exact'] and compatible['all_original_method_flags_preserved']
    assert compatible['no_consumer_or_product_build'] and not compatible['component_screen_admitted']
    assert compatible['selected'] == stage['identities']['selected']
    assert compatible['candidate'] == stage['identities']['candidate']
    assert compatible['consumers'] == stage['consumers']
    assert stage['failed_release_controls'] == compatible['failed_component_controls']
    assert len(stage['failed_release_controls']) == 13 and not stage['release_admitted']
    for label in ('root','build','models','screen'):
        proof = read(BASE/'evidence'/(label+'-closed.json'))
        assert proof['passed'] and proof['files']['analysis.json'] == pin(BASE/'evidence'/(label+'-analysis.json'))
    assert read(BASE/'evidence/root-analysis.json')['built'] == stage['identities']['selected']
    assert read(BASE/'evidence/build-analysis.json')['product'] == stage['identities']['candidate']
    assert read(BASE/'evidence/models-analysis.json')['consumers'] == stage['consumers']
    assert not read(BASE/'evidence/screen-closed.json')['admitted']
    assert not read(BASE/'evidence/screen-analysis.json')['admitted']
    memory = read(BASE/'evidence/memory-closed.json')
    assert memory['passed'] and not memory['admitted']
    for name, wanted in stage['external'].items(): assert pin(name) == wanted, name
    for name, value in stage['links'].items():
        destination = BASE/name
        assert destination.resolve().is_relative_to(BASE.resolve()) and not destination.exists()
        source = Path(value['source']); assert pin(source) == value['identity'], name
        destination.parent.mkdir(parents=True, exist_ok=True); os.link(source, destination)
    for role in ['selected','candidate']:
        folder = BASE/'runtimes'/role
        for name, wanted in stage['identities'][role].items(): assert pin(folder/name) == wanted, name
        for name, wanted in stage['consumers'].items(): assert pin(folder/(name+'.dll')) == wanted, name
        manifest = copy.deepcopy(read(BASE/'evidence/original-manifest.json'))
        manifest.update(core_sha256=stage['identities'][role]['Lokad.Onnx.dll']['sha256'],
                        data_sha256=stage['identities'][role]['Lokad.Onnx.Data.dll']['sha256'],
                        product_source=stage['labels'][role])
        (BASE/'manifests').mkdir(exist_ok=True)
        save(BASE/'manifests'/(role+'-parakeet.json'), manifest)
    payload = dict(passed=True,jobs=JOBS,limits=LIMITS,boot_time=1789634288.0,
                   **{name:stage[name] for name in ['previous_owner','identities','consumers','external','interpreter','failed_release_controls','release_admitted']},
                   scope='Full Parakeet correctness with bounded cross-product float differences: 784 arrays and 20 public clips per product/instruction mode; no performance score.',
                   files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name != 'transfer.tar.gz'})
    save(BASE/'payload.json',payload); verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__ == '__main__':
    main()
